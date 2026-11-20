import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import tensorflow as tf

from app.upload import UPLOAD_FOLDER, class_names

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'crop_disease_model.h5'
STATUS_PATH = BASE_DIR / 'models' / 'retraining_status.json'
VERSIONS_DIR = BASE_DIR / 'models' / 'candidates'
EVALUATION_DIR = BASE_DIR / 'data' / 'evaluation'
lock = threading.Lock()
status = {'state': 'idle', 'message': 'No retraining started.'}


def get_status():
    with lock:
        return status.copy()


def update_status(state, message, **details):
    global status
    with lock:
        status = {'state': state, 'message': message, **details}
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = STATUS_PATH.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(STATUS_PATH)


def image_paths(folder):
    result = []
    for index, label in enumerate(class_names):
        directory = folder / label
        if directory.is_dir():
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                    result.append((path, index))
    return result


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_images(entries):
    images, labels = [], []
    for path, label in entries:
        image = tf.io.decode_image(
            tf.io.read_file(str(path)),
            channels=3,
            expand_animations=False
        )
        image = tf.image.resize(image, (128, 128))
        image = tf.cast(image, tf.float32) / 255.0
        images.append(image.numpy())
        labels.append(label)
    return np.asarray(images, dtype='float32'), np.asarray(labels, dtype='int32')


def stratified_split(entries):
    rng = np.random.default_rng(42)
    train, validation = [], []
    for label in range(len(class_names)):
        group = [item for item in entries if item[1] == label]
        if len(group) < 5:
            continue
        order = rng.permutation(len(group))
        val_count = max(2, int(round(len(group) * 0.2)))
        validation.extend(group[int(i)] for i in order[:val_count])
        train.extend(group[int(i)] for i in order[val_count:])
    return train, validation


def accuracy(model, x, y):
    return float(np.mean(np.argmax(model.predict(x, verbose=0), axis=1) == y))


def evaluate_entries(model, entries, batch_size=32):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    if not entries:
        raise ValueError('No evaluation images found.')
    actual, predicted = [], []
    for start in range(0, len(entries), batch_size):
        batch = entries[start:start + batch_size]
        x, y = load_images(batch)
        guesses = np.argmax(model.predict(x, verbose=0), axis=1)
        actual.extend(y.tolist())
        predicted.extend(guesses.tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=list(range(len(class_names))),
        average='macro', zero_division=0
    )
    return {
        'accuracy': round(float(accuracy_score(actual, predicted)), 4),
        'macro_precision': round(float(precision), 4),
        'macro_recall': round(float(recall), 4),
        'macro_f1': round(float(f1), 4)
    }


def save_report(run_id, details):
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = VERSIONS_DIR / f'candidate_{run_id}.json'
    details['report_file'] = str(report_path.relative_to(BASE_DIR))
    report_path.write_text(json.dumps(details, indent=2), encoding='utf-8')


def retrain_worker():
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    try:
        update_status('running', 'Checking uploaded images...', run_id=run_id)
        entries = image_paths(UPLOAD_FOLDER)
        train_entries, val_entries = stratified_split(entries)
        if len(train_entries) < 10 or len(val_entries) < 4:
            raise ValueError('Not enough labelled images for retraining. Upload at least 5 images per class, with enough images for 10 training and 4 validation examples.')

        holdout_entries = image_paths(EVALUATION_DIR)
        training_hashes = {file_hash(p) for p, _ in entries}
        if any(file_hash(p) in training_hashes for p, _ in holdout_entries):
            raise ValueError('Evaluation images overlap with uploaded training images. Use separate, unseen images.')

        x_train, y_train = load_images(train_entries)
        x_val, y_val = load_images(val_entries)
        update_status('running', 'Training candidate CNN...', run_id=run_id,
                      images=len(entries), training_images=len(train_entries), validation_images=len(val_entries))
        original = tf.keras.models.load_model(MODEL_PATH, compile=False)
        baseline_val = accuracy(original, x_val, y_val)
        candidate = tf.keras.models.load_model(MODEL_PATH, compile=False)
        candidate.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = candidate.fit(x_train, y_train, validation_data=(x_val, y_val),
                                epochs=3, batch_size=16, verbose=0,
                                callbacks=[tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss', patience=1, restore_best_weights=True)])
        candidate_val = accuracy(candidate, x_val, y_val)
        details = {'run_id': run_id, 'images': len(entries),
                   'training_images': len(train_entries), 'validation_images': len(val_entries),
                   'baseline_validation_accuracy': round(baseline_val, 4),
                   'new_validation_accuracy': round(candidate_val, 4),
                   'epochs_run': len(history.history['loss']),
                   'finished_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}

        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        candidate_path = VERSIONS_DIR / f'candidate_{run_id}.h5'
        candidate.save(candidate_path)
        details['candidate_model'] = str(candidate_path.relative_to(BASE_DIR))
        details['evaluation_images'] = len(holdout_entries)
        details['internal_validation_passed'] = candidate_val > baseline_val

        per_class = {label: 0 for label in range(len(class_names))}
        for _, label in holdout_entries:
            per_class[label] += 1

        if any(count < 5 for count in per_class.values()):
            details['evaluation_note'] = 'Need at least 5 evaluation images per class.'
            save_report(run_id, details)
            update_status('needs_evaluation', 'Candidate saved but evaluation is incomplete. Original model kept.', **details)
            return

        update_status('running', 'Evaluating original and candidate models in batches...', **details)
        details['baseline_evaluation'] = evaluate_entries(original, holdout_entries)
        details['candidate_evaluation'] = evaluate_entries(candidate, holdout_entries)
        before = details['baseline_evaluation']
        after = details['candidate_evaluation']
        independent_passed = (
            after['accuracy'] >= before['accuracy'] and
            after['macro_f1'] >= before['macro_f1'] and
            (after['accuracy'] > before['accuracy'] or after['macro_f1'] > before['macro_f1'])
        )
        details['independent_evaluation_passed'] = independent_passed
        details['approved_for_manual_review'] = details['internal_validation_passed'] and independent_passed
        save_report(run_id, details)

        if details['approved_for_manual_review']:
            update_status('completed', 'Candidate passed both checks. Original model remains active pending manual review.', **details)
        else:
            update_status('rejected', 'Candidate evaluated but did not pass both checks. Original model kept.', **details)

    except Exception as error:
        update_status('failed', str(error), run_id=run_id)


def start_retraining():
    global status
    with lock:
        if status.get('state') == 'running':
            return False
        status = {'state': 'running', 'message': 'Starting retraining...'}
    threading.Thread(target=retrain_worker, daemon=True).start()
    return True

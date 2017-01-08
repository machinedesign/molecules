from machinedesign.objectives import categorical_crossentropy
from machinedesign.metrics import categorical_crossentropy as categorical_crossentropy_metric


def shifted_categorical_crossentropy(y_true, y_pred):
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    return categorical_crossentropy(y_true, y_pred)


def shifted_categorical_crossentropy_metric(y_true, y_pred):
    y_true = y_true[:, 1:, :]
    y_pred = y_pred[:, 0:-1, :]
    return categorical_crossentropy_metric(y_true, y_pred)

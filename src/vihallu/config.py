LABELS = ["no", "intrinsic", "extrinsic"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

TEXT_COLUMNS = ["context", "prompt", "response"]
ID_COLUMN = "id"
TRAIN_LABEL_COLUMN = "label"
TEST_LABEL_COLUMN = "predict_label"

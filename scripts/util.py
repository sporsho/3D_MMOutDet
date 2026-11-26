def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    iou1 = tp / (tp + fp + fn + 1e-9)
    f1 = 2 * recall * precision / (precision + recall + 1e-9)
    print(f'Class: {class_name}, Precision:{precision}, Recall: {recall}, IOU: {iou1}, F1: {f1}')
    return iou1

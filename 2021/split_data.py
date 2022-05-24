import random

folder_path = r"E:\NLP\数据集\PAN2021数据集\pan20-authorship-verification-training-small"
truth_path = folder_path + r"\pan20-authorship-verification-training-small.jsonl"
text_path = folder_path + r"\pan20-authorship-verification-training-small-truth.jsonl"
new_folder_path = folder_path + r"\data split"

for i in range(10):
    data_truth = open(truth_path, "r", encoding="utf-8")
    data_text = open(text_path, "r", encoding="utf-8")
    train_truth_path = new_folder_path + r"\pan20-authorship-verification-training-small-80-sample" + str(i + 1) + ".jsonl"
    train_text_path = new_folder_path + r"\pan20-authorship-verification-training-small-truth-80-sample" + str(i+1) +".jsonl"
    test_truth_path = new_folder_path + r"\pan20-authorship-verification-test-small-20-sample" + str(i + 1) + ".jsonl"
    test_text_path = new_folder_path + r"\pan20-authorship-verification-test-small-truth-20-sample" + str(i+1) +".jsonl"

    f1_truth = open(train_truth_path, "w", encoding="utf-8")
    f1_text = open(train_text_path, "w", encoding="utf-8")
    f2_truth = open(test_truth_path, "w", encoding="utf-8")
    f2_text = open(test_text_path, "w", encoding="utf-8")

    line_truth = data_truth.readline()
    line_text = data_text.readline()
    while line_truth:
        flag = random.random()
        if flag < 0.7:
            f1_truth.write(line_truth)
            f1_text.write(line_text)
        else:
            f2_truth.write(line_truth)
            f2_text.write(line_text)
        line_truth = data_truth.readline()
        line_text = data_text.readline()
    f1_truth.close()
    f1_text.close()
    f2_text.close()
    f2_truth.close()
    data_text.close()
    data_truth.close()
    print("loop" + str(i) + "over")


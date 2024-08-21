import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score

# 加载数据集
df = pd.read_parquet('/kaggle/input/prm800k')

# 过滤掉 rating 为 null 的样本
df = df[df['rating'].notna()]

# 将数据集分为训练集和验证集
train_df = df.iloc[:9038]
val_df = df.iloc[9038:10038]

# 定义自定义数据集类
class PRMDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        example = self.dataframe.iloc[idx]
        # 将所有特征拼接成一个文本输入
        text = " ".join([str(value) for key, value in example.items() if key not in ['rating', 'error_reason']])
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        
        # 确保 label 是一维张量，使用 `.item()` 方法获取标量值
        label = torch.tensor(example['rating'] + 1).long()  # 映射到 0, 1, 2
        inputs['labels'] = label
        
        return {key: val.squeeze(0) for key, val in inputs.items()}


# 初始化 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 创建自定义数据集
train_dataset = PRMDataset(train_df, tokenizer)
val_dataset = PRMDataset(val_df, tokenizer)

# 加载预训练的 BERT 模型并添加分类头
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# 评估函数，计算准确率
def compute_metrics(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, preds)}

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",  # 每个epoch结束后进行评估
    save_strategy="epoch",  # 每个epoch结束后保存模型
    load_best_model_at_end=True,
    logging_strategy="epoch",  # 每个epoch结束后记录日志
)

# 创建 Trainer 实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 在每个 epoch 结束后打印测试准确率
for epoch in range(training_args.num_train_epochs):
    eval_results = trainer.evaluate()
    print(f"Epoch {epoch + 1}/{training_args.num_train_epochs} - Test Accuracy: {eval_results['eval_accuracy']:.4f}")

# 在验证集上进行最终评估
final_eval_results = trainer.evaluate()
print(f"Final Test Accuracy: {final_eval_results['eval_accuracy']:.4f}")

'''
The outcome: Final Test Accuracy: 0.5030
'''

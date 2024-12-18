import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model

# Load Pretrained Chinese BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_model = TFBertModel.from_pretrained('bert-base-chinese')

# Define Custom BERT Classifier
class BertClassifier(Model):
    def __init__(self, bert):
        super(BertClassifier, self).__init__()
        self.bert = bert
        self.dropout = Dropout(0.3)
        self.classifier = Dense(1, activation='sigmoid')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Data Preparation Function
def prepare_inputs(sentences, tokenizer, max_length=64):
    input_ids, attention_masks = [], []
    for sentence in sentences:
        encoded = tokenizer.encode_plus(
            sentence, 
            add_special_tokens=True, 
            max_length=max_length, 
            pad_to_max_length=True, 
            return_attention_mask=True, 
            return_tensors='tf'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return tf.concat(input_ids, axis=0), tf.concat(attention_masks, axis=0)

# Instantiate Classifier and Compile
bert_classifier = BertClassifier(bert_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = tf.keras.losses.BinaryCrossentropy()
metric = tf.keras.metrics.BinaryAccuracy()

bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Example Sentences and Labels (Chinese)
sentences = [
    "我很開心今天的天氣很好。", "這個工作讓我感到很滿足。",
    "我討厭交通堵塞。", "這次的計畫失敗讓我很失望。",
    "我喜歡看電影。", "這件事情讓我感到壓力很大。",
    "週末和家人去郊遊，我非常快樂。", "我覺得這次會議很浪費時間。"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]

# Train Model with Increased Epochs
input_ids, attention_masks = prepare_inputs(sentences, tokenizer)
labels_tensor = tf.convert_to_tensor(labels)
bert_classifier.fit([input_ids, attention_masks], labels_tensor, epochs=5, batch_size=2)  # Increased epochs to 10

# Predict New Sentences
test_sentences = ["我喜歡去旅行", "我對這次會議感到非常厭倦"]
test_input_ids, test_attention_masks = prepare_inputs(test_sentences, tokenizer)
predictions = bert_classifier.predict([test_input_ids, test_attention_masks])

# Output Predictions
for i, sentence in enumerate(test_sentences):
    sentiment = "Positive" if predictions[i][0] > 0.5 else "Negative"
    print(f"Test sentence: {sentence}, Predicted sentiment: {sentiment}")

from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import numpy as np
import h5py
import os

def predict_resume_category(resume_text):
    max_resume_len = 200

    input_ids = Input(shape=(max_resume_len,), dtype=tf.int32, name='input_ids')
    attention_masks = Input(shape=(max_resume_len,), dtype=tf.int32, name='attention_mask')

    tokenizer = AutoTokenizer.from_pretrained("manishiitg/distilbert-resume-parts-classify")
    bert_model = TFDistilBertForSequenceClassification.from_pretrained("manishiitg/distilbert-resume-parts-classify", from_pt=True)

    word_embeddings = bert_model(input_ids, attention_mask=attention_masks)[0]

    output = Flatten()(word_embeddings)
    output = Dense(units=1024, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.25)(output)
    output = Dense(units=512, activation='relu')(output)
    output = Dropout(0.25)(output)
    output = Dense(units=256, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.25)(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dropout(0.25)(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=25, activation='softmax')(output)

    model = Model(inputs=[input_ids, attention_masks], outputs=output)

    file_path = os.path.dirname(__file__)
    print(file_path)

    filename = '/home/prazzwalthapa/Desktop/NLP_Project_Jobs/Application/backend/matcher_api/resume_matcher/Bert_Model/resume_parser.h5'


    model.load_weights(filename)

    print("loaded")
    encoded_resume = tokenizer(text=resume_text,
                               add_special_tokens=True,
                               padding=True,
                               truncation=True,
                               max_length=max_resume_len,
                               return_tensors='tf',
                               return_attention_mask=True,
                               return_token_type_ids=False,
                               verbose=1)

    predictions = model.predict({'input_ids': encoded_resume['input_ids'], 'attention_mask': encoded_resume['attention_mask']})


    return np.argmax(predictions, axis=1)



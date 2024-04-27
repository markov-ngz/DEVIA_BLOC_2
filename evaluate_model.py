import evaluate
import numpy as np 
from datetime import datetime

def compute_bleu(test_data, model ,tokenizer):
    """
    source code  : https://www.kaggle.com/code/fathykhader/fine-tuning-a-translation-model-tensorflow
    """
    all_label=[]
    predictions=[]
    count=0
    for batch,labels in test_data:

        print(f"[{datetime.now().strftime('%H:%M:%S')}] :  Beginning Process of batch n°{count}")  

        prediction=model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=128,
            )
        prediction=tokenizer.batch_decode(prediction,skip_special_tokens=True)

        label=labels.numpy()
        label=np.where(label!=-100,label,tokenizer.pad_token_id)
        label=tokenizer.batch_decode(label,skip_special_tokens=True)

        prediction=[i.strip() for i in prediction]
        label=[i.strip() for i in label]

        all_label.extend(label)
        predictions.extend(prediction)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] :  Batch n° {count} processed")  
        count+=1
    return all_label,predictions

def get_bleu_score(ds,model,tokenizer):

    metric=evaluate.load('sacrebleu')
    all_label,predictions = compute_bleu(ds, model, tokenizer)
    bleu_score = metric.compute(predictions=predictions, references=all_label)

    return bleu_score
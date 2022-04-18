# Training notes

### CUDA error: out of memory
Solution: reducing batch size (bs in dataloaders()) to 8

### GPU RAM peak on start 
?

### Single cycle of training
5 minutes
Two cycles - 24 minutes

| epoch  | train_loss | valid_loss |time|
| ------ | ---------- | ---------- | ---|
|0	     | 0.067157	  | 0.029004   | 20:55|
|1       | 0.021789	  | 0.018149   | 03:27|


### Don't use lambdas
Hard to export and resue them

### Correct loss function
?

### save vs export
https://forums.fast.ai/t/when-would-you-use-learn-save-vs-learn-export/81837

### save pips and models on colab
?
import os
from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory('C:\\Users\\Admin\\Desktop\\FinalProject\\pokemon')
model_trainer.trainModel(num_objects=5, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True)

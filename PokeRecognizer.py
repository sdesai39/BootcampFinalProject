from imageai.Prediction.Custom import CustomImagePrediction
import os

execution_path = os.getcwd()

prediction = CustomImagePrediction()
prediction.setModelTypeAsResNet()
prediction.setModelPath("C:\\Users\\Admin\\Desktop\\FinalProjectTest\\model_ex-010_acc-0.713693.h5")
prediction.setJsonPath("C:\\Users\\Admin\\Desktop\\FinalProjectTest\\model_class.json")
prediction.loadModel(num_objects=5)

predictions, probabilities = prediction.predictImage("C:\\Users\\Admin\\Desktop\\FinalProjectTest\\StaryuTest.jpg", result_count=3)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
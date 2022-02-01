import resnet18


def Generator():
    return resnet18.Feature_base()

def Standard_Classifier(args):
    return resnet18.Standard_Predictor(args)


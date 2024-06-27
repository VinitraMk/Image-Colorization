import matplotlib.pyplot as plt
from common.utils import get_exp_params

class Visualization:

    def __init__(self, model_info, model_history = {}):
        self.bestm_valloss = model_info['results']['valloss']
        self.bestm_trloss = model_info['results']['trloss']
        self.exp_params = get_exp_params()
        num_epochs = self.exp_params['train']['num_epochs']
        self.bestm_tlh = model_info['results']['trlosshistory'][-num_epochs:]
        self.bestm_vlh = model_info['results']['vallosshistory'][-num_epochs:]
        self.model_history = model_history if model_history != {} else None

    def __plot_loss_history(self):
        num_epochs = self.exp_params["train"]["num_epochs"]
        plt.clf()
        print("\nModel results\n\n")
        plt.plot(list(range(num_epochs)), self.bestm_tlh, color="red", label="Best model training loss history")
        plt.plot(list(range(num_epochs)), self.bestm_vlh, color="orange", label="Best model validation loss history")
        plt.title("Model loss history")
        plt.legend()
        plt.show()
        print("\nModel Training Loss:", self.bestm_trloss)
        print("Model Validation Loss:", self.bestm_valloss)

    def __get_performance_metrics(self):
        k = self.exp_params['train']['k']
        if self.model_history != None:
            avg_valloss = 0.0
            avg_trloss = 0.0
            print("\nFold\tTraining Loss\tValidation Loss\tValidation Accuracy")
            for fi in self.model_history:
                print(f"{fi}\t{self.model_history[fi]['trloss']}\t{self.model_history[fi]['valloss']}\t{self.model_history[fi]['valacc']}")
                avg_valloss += self.model_history[fi]['valloss']
                avg_trloss += self.model_history[fi]['trloss']
            avg_valloss/=k
            avg_trloss/=k
            print("\nAverage Performance Metrics of Model")
            print("Average Training Loss", avg_trloss)
            print("Average Validation Loss", avg_valloss)

    def get_results(self):
        self.__plot_loss_history()
        if self.exp_params['train']['val_split_method'] == 'k-fold':
            self.__get_performance_metrics()


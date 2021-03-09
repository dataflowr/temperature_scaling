import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.nn import functional as F


class ModelWithIsotonic(nn.Module):
    """
    A thin decorator, which wraps a model with isotonic regression
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, n_class=100):
        super(ModelWithIsotonic, self).__init__()
        self.model = model
        self.n_class = n_class
        self.regressors = [IsotonicRegression(out_of_bounds='clip') for _ in range(n_class)]

    def forward(self, input):
        """
        Returns the rescaled probabilities after the isotonic regressors have been trained on valid set.
        """

        logits = self.model(input)
        probs_val = F.softmax(logits, 1)
        for k in range(self.n_class):
            tmp = self.regressors[k].predict(probs_val[:,k].detach().cpu().numpy())
            probs_val[:, k] = torch.tensor(tmp).cuda()
        return probs_val

    def isotonic_regression(self, logits, labels):
        """
        Perform Isotonic regression in a 1-vs-all way. Only used to calibrate the regressors.
        """
        probs_val = F.softmax(logits, 1)
        n_class = logits.shape[1]

        probs_val = probs_val.cpu().detach().numpy()
        for k in range(n_class):

            # 1 is class k and 0 other classes
            y_cal = (labels == k)
            y_cal = y_cal.detach().cpu().numpy()

            #Calibrate model
            self.regressors[k].fit(probs_val[:,k], y_cal)

            probs_val[:,k] = self.regressors[k].predict(probs_val[:,k])


        return torch.tensor(probs_val).cuda()

    # Sets the Isotonic scaling parameter. The name is set_temperature for convenience.
    def set_temperature(self, valid_loader):
        """
        Tune the isotonic regressors for each class (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.NLLLoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL before temperature scaling
        before_temperature_nll = nll_criterion(F.log_softmax(logits,1), labels).item()
        print('Before temperature - NLL: %.3f' % (before_temperature_nll))

        # Calculate NLL after temperature scaling
        after_temperature_nll = nll_criterion(torch.log(self.isotonic_regression(logits, labels)), labels).item()
        print('After temperature - NLL: %.3f' % (after_temperature_nll))

        return self


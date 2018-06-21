import numpy
import scipy.special

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_func = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

def train_neural_network(neural_network, train_file):
    for ln in train_file:
        all_values = ln.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(neural_network.onodes) + 0.01
        targets[int(all_values[0])] = 0.99
        neural_network.train(inputs, targets)
    return neural_network


def test_neural_network(neural_network, test_file):
    scorecard = []
    for ln in test_file:
        all_values = ln.split(',')
        correct_label = int(all_values[0])
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = neural_network.query(inputs)
        label = numpy.argmax(outputs)
        # print(label, "network's answer")

        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    return scorecard

if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("start training")
    with open("mnist_train.csv") as train_f:
        new_nn = train_neural_network(nn, train_f)

    print("start testing")
    with open("mnist_test.csv") as test_f:
        result = test_neural_network(new_nn, test_f)
    scorecard_array = numpy.asarray(result)
    print("perfornamce = ", scorecard_array.sum() / scorecard_array.size)

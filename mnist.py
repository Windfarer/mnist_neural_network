import numpy
from neural_network import NeuralNetwork

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

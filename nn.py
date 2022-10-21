import numpy
import math


class ComputeNode:
    def __init__(self, type, id, *children):
        self.type = type
        self.id = id
        self.children = []
        self.children.extend(children)
        self.value = None
        self.grad = None


class Graph:
    def __init__(self):
        self.nodes = []

    def var(self, value):
        a = ComputeNode('var', len(self.nodes))
        a.value = value
        self.nodes.append(a)
        return a

    def add(self, a, b):
        v = ComputeNode('add', len(self.nodes), a, b)
        self.nodes.append(v)
        return v

    def mul(self, a, b):
        v = ComputeNode('mul', len(self.nodes), a, b)
        self.nodes.append(v)
        return v

    def logistic(self, a):
        v = ComputeNode('logistic', len(self.nodes), a)
        self.nodes.append(v)
        return v

    def logsoftmax(self, a):
        v = ComputeNode('logsoftmax', len(self.nodes), a)
        self.nodes.append(v)
        return v

    def cross_entropy(self, pred, gold):
        logprob = self.logsoftmax(pred)
        v = ComputeNode('cross-entropy', len(self.nodes), logprob, gold)
        self.nodes.append(v)
        return v


def forward(graph):
    for n in graph.nodes:
        if n.type == 'var':
            # do nothing
            pass
        elif n.type == 'add':
            n.value = n.children[0].value + n.children[1].value
        elif n.type == 'mul':
            n.value = n.children[0].value @ n.children[1].value
        elif n.type == 'logistic':
            n.value = 1.0 / (1.0 + numpy.exp((-1) * n.children[0].value))
        elif n.type == 'logsoftmax':
            # log sum exp trick
            m = max(n.children[0].value)
            Z = math.log(sum(numpy.exp(n.children[0].value - m))) + m
            n.value = n.children[0].value - Z
        elif n.type == 'cross-entropy':
            n.value = (-1) * numpy.dot(n.children[0].value, n.children[1].value)
        else:
            print('unknown node type: {}'.format(n.type))


def init_grad(n):
    if n.grad is None:
        n.grad = numpy.zeros_like(n.value)


def backward(graph):
    for n in reversed(graph.nodes):
        if n.type == 'var':
            # do nothing
            pass
        elif n.type == 'add':
            init_grad(n.children[0])
            init_grad(n.children[1])
            n.children[0].grad += n.grad
            n.children[1].grad += n.grad
        elif n.type == 'mul':
            init_grad(n.children[0])
            init_grad(n.children[1])
            n.children[0].grad += n.grad @ n.children[1].value.T
            n.children[1].grad += numpy.outer(n.children[0].value, n.grad)
        elif n.type == 'logistic':
            init_grad(n.children[0])
            n.children[0].grad += n.grad * n.value * (1 - n.value)
        elif n.type == 'logsoftmax':
            init_grad(n.children[0])
            n.children[0].grad += n.grad - sum(n.grad) * numpy.exp(n.value)
        elif n.type == 'cross-entropy':
            init_grad(n.children[0])
            n.children[0].grad += (-1) * n.children[1].value
        else:
            print('unknown node type: {}'.format(n.type))


def init_vec(d):
    """
    Glorot initialization
    """
    u = numpy.random.rand(d) * 2 - 1
    return u * 6 / math.sqrt(d)


def init_mat(rows, cols):
    """
    Glorot initialization
    """
    u = numpy.random.rand(rows, cols) * 2 - 1
    return u * 6 / math.sqrt(rows + cols)


def one_hot(i, n):
    v = numpy.zeros(n)
    v[i] = 1
    return v


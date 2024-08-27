import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import ltn
import itertools
import matplotlib.pyplot as plt


num_objects = 8
object_dim = 8

object_pairs = torch.Tensor(list(itertools.permutations(range(num_objects), r=2))).int()
object_order_relations = object_pairs[:, 0] < object_pairs[:, 1]
neg_pairs = object_pairs[object_order_relations == False]
pos_pairs = object_pairs[object_order_relations]

pos_chain = torch.Tensor([[i, i + 1] for i in range(num_objects - 1)]).int()


one_hot = torch.eye(num_objects)
x1 = ltn.Variable("x1", one_hot)
x2 = ltn.Variable("x2", one_hot)
x3 = ltn.Variable("x3", one_hot)

c = [ltn.Constant(features, trainable=False) for features in one_hot]

Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Equiv = ltn.Connective(
    ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach())
)
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")

# Forall_strong = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=10), quantifier="f")
Forall_strong = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")


def train(
    model,
    epoch_steps=50,
    epochs=1000,
    reflective=True,
    train_chain=True,
    transitive=True,
    antisymmetric=True,
    neg_sample=None,
    pos_sample=None,
    pos_pairs=None,
    neg_pairs=None,
    lr=0.001,
    switch=False,
):
    heatmap_data = []
    query = model(x1, x2).value.detach().numpy()
    heatmap_data.append(query)
    # by default, SatAgg uses the pMeanError
    sat_agg = ltn.fuzzy_ops.SatAgg()

    # we need to learn the parameters of the predicate C
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    last_loss = np.inf

    for epoch in range(epochs):
        optimizer.zero_grad()
        fuzzy_theory = []

        if pos_pairs is not None:
            fuzzy_theory += [(model(c[i.item()], c[j.item()])) for (i, j) in pos_pairs]
        elif pos_sample is not None:
            fuzzy_theory += [model(c[i.item()], c[j.item()]) for (i, j) in pos_sample]

        if neg_pairs is not None:
            fuzzy_theory += [
                Not(model(c[i.item()], c[j.item()])) for (i, j) in neg_pairs
            ]
        elif neg_sample is not None:
            fuzzy_theory += [
                Not(model(c[i.item()], c[j.item()])) for (i, j) in neg_sample
            ]

        if train_chain:
            fuzzy_theory += [model(c[i.item()], c[j.item()]) for (i, j) in pos_chain]
        if reflective:
            fuzzy_theory.append(Forall(x1, model(x1, x1)))
        if transitive:
            fuzzy_theory.append(
                Forall_strong(
                    [x1, x2, x3],
                    Implies(And(model(x1, x2), model(x2, x3)), model(x1, x3)),
                )
            )
        if antisymmetric:
            fuzzy_theory.append(
                Forall(
                    [x1, x2],
                    Implies(model(x1, x2), Not(model(x2, x1))),
                    cond_vars=[x1, x2],
                    cond_fn=lambda x, y: ~torch.eye(num_objects, dtype=bool),
                )
            )

        loss = 1.0 - sat_agg(*fuzzy_theory)
        loss.backward()
        optimizer.step()

        # if epoch%200 == 0:
        if epoch % epoch_steps == 0:
            loss = loss.item()
            sat_score = 1 - loss
            print("Epoch %d: Sat Level %.3f " % (epoch, sat_score))
            query = model(x1, x2).value.detach().numpy()
            heatmap_data.append(query)
            if 1 - loss > 0.995 or np.allclose(
                loss, last_loss, atol=1e-04, equal_nan=False
            ):
                if not switch:
                    break
                else:
                    print()
                    print(epoch / epoch_steps)
                    switch = False
                    antisymmetric = True
                    # Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=6), quantifier="f")

            last_loss = loss

    print("Training finished at Epoch %d with Sat Level %.3f" % (epoch, 1 - loss))

    return heatmap_data

import ltn

Godel = dict(
    Not=ltn.Connective(ltn.fuzzy_ops.NotGodel()),
    And=ltn.Connective(ltn.fuzzy_ops.AndMin()),
    Or=ltn.Connective(ltn.fuzzy_ops.OrMax()),
    Implies=ltn.Connective(ltn.fuzzy_ops.ImpliesGodel()),
)

KleeneDienes = dict(
    Not=ltn.Connective(ltn.fuzzy_ops.NotGodel()),
    And=ltn.Connective(ltn.fuzzy_ops.AndMin()),
    Or=ltn.Connective(ltn.fuzzy_ops.OrMax()),
    Implies=ltn.Connective(ltn.fuzzy_ops.ImpliesKleeneDienes()),
)

Goguen = dict(
    Not=ltn.Connective(ltn.fuzzy_ops.NotStandard()),
    And=ltn.Connective(ltn.fuzzy_ops.AndProd()),
    Or=ltn.Connective(ltn.fuzzy_ops.OrProbSum()),
    Implies=ltn.Connective(ltn.fuzzy_ops.ImpliesGoguen()),
)

Reichenbach = dict(
    Not=ltn.Connective(ltn.fuzzy_ops.NotStandard()),
    And=ltn.Connective(ltn.fuzzy_ops.AndProd()),
    Or=ltn.Connective(ltn.fuzzy_ops.OrProbSum()),
    Implies=ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach()),
)

Luk = dict(
    Not=ltn.Connective(ltn.fuzzy_ops.NotStandard()),
    And=ltn.Connective(ltn.fuzzy_ops.AndLuk()),
    Or=ltn.Connective(ltn.fuzzy_ops.OrLuk()),
    Implies=ltn.Connective(ltn.fuzzy_ops.ImpliesLuk()),
)

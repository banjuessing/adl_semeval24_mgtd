import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


def get_augmentation(augmodel_path, 
                     all=True, 
                     synonym=True, antonym=False, swap=False, spelling=False, 
                     word2vec=False, contextual=False):
    ops = []

    if synonym:
        ops.append(naw.SynonymAug(aug_src='wordnet'))

    if antonym:
        ops.append(naw.AntonymAug())

    if swap:
        ops.append(naw.RandomWordAug(action='swap'))

    if spelling:
        ops.append(naw.SpellingAug())

    if word2vec:
        ops.append(naw.WordEmbsAug(
            model_type='word2vec', 
            model_path=augmodel_path+'GoogleNews-vectors-negative300.bin', 
            action='substitute'))
        
    if contextual:
        ops.append(naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action='substitute'))
    
    if all:
        aug = naf.Sequential(ops)
    else:
        aug = naf.Sometimes(ops)

    return aug
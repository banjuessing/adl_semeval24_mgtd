import nlpaug.augmenter.word as naw
import nlpaug.flow as naf


def get_augmentation(augmodel_path, 
                     all=True, 
                     synonym=True, antonym=False, swap=False, spelling=False, 
                     word2vec=False):
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
    
    if all:
        aug = naf.Sequential(ops)
    else:
        aug = naf.Sometimes(ops)

    return aug


if __name__ == "__main__":
    text = ["Phil Read (born 20 October 1979) was an Australian rules footballer for the AFL's West Coast Eagles and Melbourne Football Club.\n\nRead was educated at Wesley College, Perth.",
            "In the context of imperial China, the three major religions, Buddhism, Confucianism, and Taoism, did not have any specific relationships or interactions with each other. They were separate and distinct belief systems."]
    
    aug = get_augmentation("./augmodel/", all=True, synonym=True, swap=True)

    augmented = aug.augment(text)
    print(augmented)
    
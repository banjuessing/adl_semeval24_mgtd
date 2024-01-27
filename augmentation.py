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


if __name__ == "__main__":
    text = ["Phil Read (born 20 October 1979) was an Australian rules footballer for the AFL's West Coast Eagles and Melbourne Football Club.\n\nRead was educated at Wesley College, Perth.\n\nWest Coast Eagles \nDrafted by the Eagles, Read kicked 12 goals in 21 games in his rookie season. During the \"Demolition Derby\" in Round 21, 2000, Read was struck numerous times during an attempted fight with Fremantle's Dale Kickett. This started a melee involving a number of players which led to Kickett being suspended for a total of 9 weeks and Read for two with a hefty fine for multiple wrestling offences.\n\nPersistent injury problems forced him to miss the entire 2001 and 2002 seasons before returning in 2003 to play 11 games before being delisted at the end of the season.\n\nMelbourne career \nSelected by the Demons in the 2004 Pre-Season draft Read played 21 games in his first season for his new club but hamstring injuries struck again over the next two years and he was restricted to 8 games in 2005.\n\nRead reappeared in the Melbourne seniors midway through the 2006 season but remained a bit-part player for the rest of the year. Despite a string of best-on-ground honors playing for Melbourne's reserve club Sandringham (including a 46 possession game) he was unable to break back into the Melbourne squad for the finals. Read was voted best on ground in Sandringham's Victorian Football League 2006 Grand Final victory over Geelong.\n\nRead was delisted by Melbourne at the end of the 2006 season. He signed with Geelong reserves for the 2007 season.\n\nAfter signing with Geelong reserves, Phillip Read was then linked to the Subiaco Lions in the WAFL for the 2008 season. Phil Read was crucial in Subiaco's third consecutive Premiership.\n\nReferences\n\nExternal links\n\nDemon Wiki profile\n\n1979 births\nLiving people\nMelbourne Football Club players\nWest Coast Eagles players\nEast Fremantle Football Club players\nSandringham Football Club players\nAustralian rules footballers from Western Australia\nPeople educated at Wesley College, Perth",
            "In the context of imperial China, the three major religions, Buddhism, Confucianism, and Taoism, did not have any specific relationships or interactions with each other. They were separate and distinct belief systems, each with its own history, teachings, and practices.\n\nBuddhism:\n\nBuddhism originated in India and was introduced to China in the 1st century AD. It quickly gained popularity and was widely adopted by the ruling class and the common people. In China, Buddhism was seen as a foreign religion and was often viewed with suspicion and suspicion. The ruling class, in particular, was concerned about the political implications of Buddhism and its potential threat to the established social order.\n\nConfucianism:\n\nConfucianism was the dominant belief system in China and was deeply ingrained in Chinese society. It was based on the teachings of Confucius, a philosopher who lived in the 6th century BC. Confucianism emphasized the importance of social order, family values, and the maintenance of the status quo.\n\nTaoism:\n\nTaoism was a native Chinese belief system that emerged in the 4th century BC."]
    
    aug = get_augmentation("./augmodel/", all=True, synonym=True, word2vec=True)

    augmented = aug.augment(text)
    print(augmented)
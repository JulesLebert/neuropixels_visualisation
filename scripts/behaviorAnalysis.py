# Goal with this script is to first serve as a wrapper for Jules's behavior analysis, but
# then to allow me to implement my own. I also took his data-loading code.


from pathlib import Path

import pandas as pd
from instruments import behaviouralAnalysis
from instruments.io.BehaviourIO import BehaviourDataSet
def main():
    #behaviorFilepath = Path('Y:/Data/Behaviour/Ferret/')
    behaviorFilepath = Path('C:/Users/Soraya/Dropbox/Data/')
    ferrets = {
        #'F2301_Boule': ['01-08-2024','16-10-2024'],
        #'F2105_Clove': ['01-08-2024', '16-10-2024'],
        'F2302_Challah': ['01-08-2024', '16-10-2024']
    }
    allData = pd.DataFrame()
    for ferret in ferrets:
        startDate = ferrets[ferret][0]
        finishDate = ferrets[ferret][1]
        ds = BehaviourDataSet(startDate=startDate, finishDate=finishDate, ferrets=ferret,filepath=behaviorFilepath)
        currFerretData = ds.load()
        currFerretData['ferret'] = ferret
        allData = pd.concat([allData, currFerretData], ignore_index=True)

    behaviouralAnalysis.createWeekBehaviourFigs(allData) ###

if __name__ == '__main__':
    main()
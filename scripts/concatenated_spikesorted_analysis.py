### This is mostly copied from Jules's instruments. I copied instead of doing a git clone because I want to edit it heavily.
### lots of it is probably broken because, as you can see, I commented out all of the instruments dependencies.


from pathlib import Path
from dataclasses import dataclass
import numpy as np
import scipy as sp
import cupyx as cpx
import cupyx.scipy.signal as cpx_sig
from scipy.signal import resample, butter
from sklearn.linear_model import Ridge
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import colormaps

import seaborn as sns
from tqdm import tqdm
import pickle
from probeinterface.plotting import plot_probe
from neuropixels_visualisation.config import bhv_fs, np_fs

import naplib as nl

# from instruments.io.phyconcatrecio import PhyConcatRecIO
# from instruments.helpers.extract_helpers import load_bhv_data
# from instruments.config import warpDataPath, figure_output
from instruments.helpers import util
from instruments.helpers import extract_helpers as inst_extract_helpers

from neuropixels_visualisation.helpers.neural_analysis_helpers import NeuralDataset, concatenatedNeuralData, align_times
from neuropixels_visualisation.config import neuropixelDataPath



def apply_filter(df, filter,window = [-1,2]):
    if filter == 'Target trials':
        return df[df['catchTrial'] != 1]
    if filter == 'Catch trials':
        return df[df['catchTrial'] == 1]
    if filter == 'Level cue':
        return df[df['currAtten'] > 0]
    if filter == 'No Level Cue':
        return df[df['currAtten'] == 0]
    if filter == 'Non Correction Trials':
        return df[df['correctionTrial'] == 0]
    if filter == 'Correction Trials':
        return df[df['correctionTrial'] == 1]
    if filter == 'Sound Right':
        return df[df['side'] == 1]
    if filter == 'Sound Left':
        return df[df['side'] == 0]
    if filter == 'Correct':
        return df[df['correct'] == 1]
    if filter == 'Noiseless':
        return df[df['currNoiseAtten'] == 120]
    if filter == 'Noiseful':
        return df[df['currNoiseAtten'] < 100]
    if filter == 'TrialDoesNotEnd':
        # only include trials which end after the window is over. For "trial start" windows.
        return df[df['centreRelease'] > window[1]]
    if filter == 'TargetNotInWindow':
        # only includes trials for which the target does not appear in the window selected. For "trial start" windows.
        return df[df['targTimes'] > window[1]]
    if filter == 'TargetAtLeast2SecondsIn':
        # only includes trials for which the target appears after 2 seconds. For "target onset" windows.
        return df[df['targTimes'] > 2]
    else:
        return f'Filter "{filter}" not found'


@dataclass
class concatenatedAnalysis(concatenatedNeuralData):
    # class concatenatedNeuralData:
    #     dp: str
    #     currNeuralDataPath: str = warpDataPath
    #     datatype: str = 'warp' # Either 'neuropixel' or 'warp'
    #     overwrite_pkl: bool = False

    #     def load(self):
    #         print("Loading data from:", self.dp)
    #         phy_folder = Path(self.dp)
    #         self.evtypes = {'Trial Start': 'startTrialLick', 'Target Time': 'absoluteTargTimes',
    #             'Release Time': 'absoluteRealLickRelease'}

    #         if (phy_folder / 'blocks.pkl').exists() and not self.overwrite_pkl:
    #             with open(phy_folder / 'blocks.pkl', 'rb') as f:
    #                 self.blocks = pickle.load(f)
    #         else:
    #             self.reader = PhyConcatRecIO(dirname = phy_folder,
    #                                 currNeuralDataPath=self.currNeuralDataPath,
    #                                 datatype=self.datatype)
    #             self.blocks = self.reader.read()

    #             for seg in self.blocks[0].segments:
    #                 if seg.annotations['bhv_file'] is not None:
    #                     seg.df_bhv = load_bhv_data(seg.annotations['bhv_file'])
    #                 else:
    #                     seg.df_bhv = None

    #             with open(phy_folder / 'blocks.pkl', 'wb') as f:
    #                 print('save pickle blocks file in:', phy_folder / 'blocks.pkl')
    #                 pickle.dump(self.blocks, f)

    def align_neuron_to_ev_for_multi_neuron_analyses(self,
                           cluster_id,
                           evtype,
                           filter_trials={},
                           window=[-1, 2],
                           fr_threshold=1):
        ### this function is like align_neuron_to_ev, but it stores an absolute trial and session number instead of a relative trial number to make sure that the neurons are aligned trial by trial in this analysis.
        aligned_spikes = []
        for sessionNumber,seg in enumerate(self.blocks[0].segments): ### A "segment" is a session, so we are scrolling through sessions.
            unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == cluster_id][0] ### this grabs the "spiketrain" corresponding to the cluster inputted into this function. I believe this is a list of all the spike times.

            # Only keep unit for this session if firing rate > 0.5/s
            # if statistics.mean_firing_rate(unit) < 0.5:
            #     continue

            if seg.df_bhv is None: ### I forget exactly what this means but I imagine it is related to the behavior (bhv) file.
                continue

            filtered_bhv = seg.df_bhv.copy() # The bhv contains all of the data relating to each trial of the session... The filtered file is the file after filtering (obviously). The filter is not a DSP type but a variable type. Usually you will have gotten rid of the "target louder than ref" trials. At least I thought that was the case but the trial number column seems to be including everything. Looking down a bit, I see that the filtering has not been performed yet.
            if len(filter_trials) > 0:
                for filter in filter_trials:
                    filtered_bhv = apply_filter(filtered_bhv, filter,window)

            # Get event times
            if filtered_bhv.shape[0] > 0:
                ev_times = filtered_bhv[self.evtypes[evtype]].to_numpy()

                # Get spike times
                seg_aligned_spikes = align_times(unit.times.magnitude, ev_times, window)
                ### new stuff
                temp_indices = np.unique(seg_aligned_spikes[:,0])
                true_indices = filtered_bhv.index[[int(x) for x in np.unique(seg_aligned_spikes[:,0])]].to_numpy()
                temp_seg_aligned_spikes = seg_aligned_spikes
                for i,temp_index in enumerate(temp_indices):
                    seg_aligned_spikes[temp_seg_aligned_spikes[:,0] == temp_index,0] = true_indices[i]
                clusterVector = cluster_id*np.ones((np.shape(seg_aligned_spikes)[0],1))
                sessionVector = (sessionNumber+1)*np.ones((np.shape(seg_aligned_spikes)[0],1)) # "sessionNumber" should be safe to use here because we continue if the cell doesn't have any spikes in the session or w/e.  Meaning that the absolute value of the seg is always incremented by the same amount.
                seg_aligned_spikes = np.append(seg_aligned_spikes, clusterVector, 1)
                seg_aligned_spikes = np.append(seg_aligned_spikes, sessionVector, 1)
                if len(aligned_spikes) == 0:
                    aligned_spikes = seg_aligned_spikes
                else:
                    aligned_spikes = np.concatenate((aligned_spikes, seg_aligned_spikes)) # just concatenate, trusting that the session count will keep the different trials seperate.
            else:
                lauch='haha'
        return aligned_spikes

    def align_neuron_to_ev(self,
                           cluster_id,
                           evtype,
                           filter_trials={},
                           window=[-1, 2],
                           fr_threshold=1):

        aligned_spikes = []
        for seg in self.blocks[0].segments: ### A "segment" is a session, so we are scrolling through sessions.
            unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == cluster_id][0] ### this grabs the "spiketrain" corresponding to the cluster inputted into this function. I believe this is a list of all the spike times.

            # Only keep unit for this session if firing rate > 0.5/s
            # if statistics.mean_firing_rate(unit) < 0.5:
            #     continue

            if seg.df_bhv is None: ### I forget exactly what this means but I imagine it is related to the behavior (bhv) file.
                continue

            filtered_bhv = seg.df_bhv.copy() # The bhv contains all of the data relating to each trial of the session... The filtered file is the file after filtering (obviously). The filter is not a DSP type but a variable type. Usually you will have gotten rid of the "target louder than ref" trials. At least I thought that was the case but the trial number column seems to be including everything. Looking down a bit, I see that the filtering has not been performed yet.
            if len(filter_trials) > 0:
                for filter in filter_trials:
                    filtered_bhv = apply_filter(filtered_bhv, filter,window)

            # Get event times
            ev_times = filtered_bhv[self.evtypes[evtype]].to_numpy()

            # Get spike times
            seg_aligned_spikes = align_times(unit.times.magnitude, ev_times, window)
            if len(aligned_spikes) == 0:
                aligned_spikes = seg_aligned_spikes
            else:
                seg_aligned_spikes[:, 0] = seg_aligned_spikes[:, 0] + aligned_spikes[-1, 0] # this apparently numbers trials one after the other, meaning that if I want to track sessions I need to do it elsewhere.
                aligned_spikes = np.concatenate((aligned_spikes, seg_aligned_spikes))

        return aligned_spikes

        # seg.spikes_array = np.concatenate((dataset.reader._spike_clusters, dataset.reader._spike_times/seg.annotations['sampling_frequency']), axis=1, dtype=object)

    def gather_sounds_per_trial(self,
                           filter_trials={},window=[-1,2]):
        sessionNumber = 1
        all_filtered_bhv = pd.DataFrame()
        for seg in self.blocks[0].segments: ### A "segment" is a session, so we are scrolling through sessions.

            # Only keep unit for this session if firing rate > 0.5/s
            # if statistics.mean_firing_rate(unit) < 0.5:
            #     continue

            if seg.df_bhv is None: ### I forget exactly what this means but I imagine it is related to the behavior (bhv) file.
                continue
            seg.df_bhv = inst_extract_helpers.stimulus_info(seg.df_bhv)


            filtered_bhv = seg.df_bhv.copy() # The bhv contains all of the data relating to each trial of the session... The filtered file is the file after filtering (obviously). The filter is not a DSP type but a variable type. Usually you will have gotten rid of the "target louder than ref" trials. At least I thought that was the case but the trial number column seems to be including everything. Looking down a bit, I see that the filtering has not been performed yet.
            if len(filter_trials) > 0:
                for filter in filter_trials:
                    filtered_bhv = apply_filter(filtered_bhv, filter,window)
            filtered_bhv['sessionNumber'] = sessionNumber
            if all_filtered_bhv.empty:
                all_filtered_bhv = filtered_bhv
            else:
                all_filtered_bhv = pd.concat((all_filtered_bhv, filtered_bhv),axis=0) # you can recover individual sessions using sessionNumber.
            sessionNumber += 1



        return all_filtered_bhv
    def create_summary_pdf(self,
                           saveDir,
                           title='summary_pdf',doUnsorted = False,testFuncActive = False, testFuncVar = None):

        block = self.blocks[0]
        if testFuncActive:
            if testFuncVar['clustersToTest'] == 'All':
                if doUnsorted:
                    units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains]
                else:
                    #units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                    #         st.annotations['cluster_id'] in ['good', 'mua']]
                    units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                             st.annotations['group'] in ['good', 'mua']]
                self.testFun_summary_pdf(units, title, saveDir)
            else:
                units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                     st.annotations['cluster_id'] in testFuncVar['clustersToTest']]
                self.testFun_summary_pdf(units, title, saveDir)

        else:
            if doUnsorted:
                units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains]
            else:
                #units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                #     st.annotations['cluster_id'] in ['good', 'mua']]
                units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains if
                         st.annotations['group'] in ['good', 'mua']]

            self._summary_pdf(units, title, saveDir)



    def _summary_pdf(self, units, title, savedir):
        events_args = {1: {'Trial Start': ['No Level Cue'],
                           'ax_to_plot': ['A', 'D']},

                       2: {'Trial Start': ['No Level Cue', 'Sound Left'],
                           'ax_to_plot': ['B', 'E']},
                       3: {'Trial Start': ['No Level Cue', 'Sound Right'],
                           'ax_to_plot': ['C', 'F']},

                       4: {'Target Time': ['No Level Cue', 'Target trials'],
                           'ax_to_plot': ['G', 'J']},
                       5: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           'ax_to_plot': ['H', 'K']},
                       6: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           'ax_to_plot': ['I', 'L']},

                       7: {'Release Time': ['No Level Cue', 'Target trials'],
                           'ax_to_plot': ['M', 'P']},
                       8: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           'ax_to_plot': ['N', 'Q']},
                       9: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           'ax_to_plot': ['O', 'R']}, }

        # cluster_ids = [st.annotations['cluster_id'] for st in self.blocks[0].segments[0].spiketrains if st.annotations['group'] != 'noise']
        # cluster_ids = units
        saveDir = Path(savedir)
        with PdfPages(saveDir / f'{title}.pdf') as pdf:
            print(f'Saving summary figures as pdf for {self.dp}')
            for clus in tqdm(units): # So this iterates through each cluster of the ones that you either called "good or multiunit" or all of the cells if you flagged that option.
                fig = self._unit_summary_figure(clus, events_args)
                pdf.savefig(fig)
                plt.close(fig)

    def _unit_summary_figure(self, cluster_id, events_args, window=[-1, 2]):

        colors = {'Trial Start': 'red',
                  'Target Time': 'green',
                  'Release Time': 'blue'}

        mosaic = """
            ABC
            DEF
            GHI
            JKL
            MNO
            PQR
            STT
            STT
            """
        fig = plt.figure(figsize=(20, 15), dpi=300)
        ax_dict = fig.subplot_mosaic(mosaic)
        ax_keys = list(ax_dict.keys())

        for i, (fig_num, params) in enumerate(events_args.items()):
            evtype = list(params.keys())[0]
            filters = params[evtype]
            aligned_spikes = self.align_neuron_to_ev(cluster_id, evtype, filters)

            axes = [ax_dict[k] for k in params['ax_to_plot']]
            # 0, 1, 2 when i == 0, 1, 2 and 6, 7, 8 when i == 3, 4, 5

            axes[0].scatter(x=aligned_spikes[:, 1], y=aligned_spikes[:, 0], ### okay so x are all of the spike times, and y are the trials in which they occured. This plot function makes the raster plot.
                            s=1, c=colors[evtype], alpha=0.8, edgecolors='none'
                            )
            # sns.scatterplot(x = aligned_spikes[:,1], y = aligned_spikes[:,0],
            #             s=5, ax=axes[0])
            binsize = 0.05
            bins = np.arange(-1, 2, binsize) # okay so looks like we are taking 1 second before and two seconds after, hard-coded.
            # unique_trials = np.unique(aligned_spikes[:,0])
            # binned_resp = np.empty([len(unique_trials), len(bins) - 1])
            # for t, trial in enumerate(unique_trials):
            #     binned_resp[t, :] = np.histogram(aligned_spikes[:,1][aligned_spikes[:,0] == trial], bins=bins)[0]#, range=(window[0], window[1]))[0]

            # binned_resp = binned_resp / binsize

            psth, edges = np.histogram(aligned_spikes[:, 1], bins)
            psthfr = (psth / len(np.unique(aligned_spikes[:, 0]))) / binsize
            zscore = (psthfr - np.mean(psthfr)) / np.std(psthfr)

            # axes[1].hist(aligned_spikes[:,1], bins = 100,
            #     color = colors[evtype], alpha = 0.8)
            axes[1].plot(edges[:-1], zscore, color=colors[evtype], alpha=0.8,
                         linewidth=2)

            # df_binned = pd.DataFrame(binned_resp)
            # melted_data = pd.melt(df_binned.reset_index(), id_vars='index')
            # sns.lineplot(data=melted_data, x='variable', y='value', errorbar='se', ax=axes[1])
            # axes[1].set_ylim(bottom=0)
            # axes[1].set_xlim([window[0],window[1]])

            for ax in axes:
                ax.axvline(0, color=colors[evtype], linestyle='--')
                ax.set_xlabel('Time (s)')
                util.simple_xy_axes(ax)
                util.set_font_axes(ax, add_size=12)

            axes[0].set_title(f'Unit {cluster_id} {evtype} {" ".join(filters)}')

        unit = [st for st in self.blocks[0].segments[0].spiketrains if st.annotations['cluster_id'] == cluster_id][0]

        # self.plot_waveform(ax_dict['S'], unit)
        # self.plot_channel_map(ax_dict['T'], unit)

        quality = unit.annotations['group']
        fig.suptitle(f'Unit {cluster_id} {quality}')

        fig.tight_layout()

        return fig

    def testFun_summary_pdf(self, units, title, savedir): # made seperate so that I can adjust stuff independently
        whichFunction = 'InfoOverTime'#'reconstruction'
        if whichFunction == 'reconstruction':
            events_args = {
                           # 1: {'Target Time': ['No Level Cue','Target trials','Sound Right','Correct','Noiseless'], ### artificially added sound right
                           #    'ax_to_plot': ['A', 'D']}, #### temporarily only doing the one event type for debugger reasons.

                           #1: {'Trial Start': ['No Level Cue','Noiseful','Sound Left'],      ###, 'Sound Right'
                           #    'ax_to_plot': ['B', 'E']},
                            #1: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Right'],  ###, 'Sound Right'
                            1: {'Target Time': ['No Level Cue', 'Noiseless', 'Sound Right','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                                'ax_to_plot': ['A', 'D']},
                            #2: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Right'],  ###, 'Sound Right'
                            #2: {'Target Time': ['No Level Cue', 'Noiseful', 'Sound Right','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                            #    'ax_to_plot': ['B', 'E']},
                            #3: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Left'],  ###, 'Sound Right'
                            #3: {'Target Time': ['No Level Cue', 'Noiseless', 'Sound Left','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                            #    'ax_to_plot': ['C', 'F']},
                            #4: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Left'],  ###, 'Sound Right'
                            #4: {'Target Time': ['No Level Cue', 'Noiseful', 'Sound Left','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                            #    'ax_to_plot': ['G', 'J']},
                            # 1: {'Trial Start': ['No Level Cue','Noiseless','Sound Right','TargetNotInWindow'],      ###, 'Sound Right'
                            #    'ax_to_plot': ['A', 'D']},
                            # 2: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Right', 'TargetNotInWindow'],  ###, 'Sound Right'
                            #     'ax_to_plot': ['B', 'E']},
                            # 3: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Left', 'TargetNotInWindow'],  ###, 'Sound Right'
                            #     'ax_to_plot': ['C', 'F']},
                            # 4: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Left', 'TargetNotInWindow'],  ###, 'Sound Right'
                            #     'ax_to_plot': ['G', 'J']},
                           # 3: {'Trial Start': ['No Level Cue', 'Sound Right'],
                           #     'ax_to_plot': ['C', 'F']},
                           #
                           # 4: {'Target Time': ['No Level Cue', 'Target trials'],
                           #     'ax_to_plot': ['G', 'J']},
                           # 5: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           #     'ax_to_plot': ['H', 'K']},
                           # 6: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           #     'ax_to_plot': ['I', 'L']},
                           #
                           # 7: {'Release Time': ['No Level Cue', 'Target trials'],
                           #     'ax_to_plot': ['M', 'P']},
                           # 8: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                           #     'ax_to_plot': ['N', 'Q']},
                           # 9: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                           #     'ax_to_plot': ['O', 'R']},
                           }
            # cluster_ids = [st.annotations['cluster_id'] for st in self.blocks[0].segments[0].spiketrains if st.annotations['group'] != 'noise']
            # cluster_ids = units
            windowToCheck = [-1, 0] # when yu have a target/response time filter on, it's important to pay attention to this
            self.reconstructionAnalysis(units, events_args,window=windowToCheck) # "units" inputted instead of cluster, because this code I am currently working on wants all units for one trial.
        elif whichFunction == 'InfoOverTime':
            events_args = {
                # 1: {'Target Time': ['No Level Cue','Target trials','Sound Right','Correct','Noiseless'], ### artificially added sound right
                #    'ax_to_plot': ['A', 'D']}, #### temporarily only doing the one event type for debugger reasons.

                # 1: {'Trial Start': ['No Level Cue','Noiseful','Sound Left'],      ###, 'Sound Right'
                #    'ax_to_plot': ['B', 'E']},
                1: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Right','TargetNotInWindow'],  ###, 'Sound Right'
                #1: {'Target Time': ['No Level Cue', 'Noiseless', 'Sound Right', 'Correct', 'TargetAtLeast2SecondsIn'],
                    ###, 'Sound Right'
                    'ax_to_plot': ['A', 'D']},
                2: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Right','TargetNotInWindow'],  ###, 'Sound Right'
                # 2: {'Target Time': ['No Level Cue', 'Noiseful', 'Sound Right','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                    'ax_to_plot': ['B', 'E']},
                 3: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Left','TargetNotInWindow'],  ###, 'Sound Right'
                # 3: {'Target Time': ['No Level Cue', 'Noiseless', 'Sound Left','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                    'ax_to_plot': ['C', 'F']},
                 4: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Left','TargetNotInWindow'],  ###, 'Sound Right'
                # 4: {'Target Time': ['No Level Cue', 'Noiseful', 'Sound Left','Correct','TargetAtLeast2SecondsIn'],  ###, 'Sound Right'
                    'ax_to_plot': ['G', 'J']},
                # 1: {'Trial Start': ['No Level Cue','Noiseless','Sound Right','TargetNotInWindow'],      ###, 'Sound Right'
                #    'ax_to_plot': ['A', 'D']},
                # 2: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Right', 'TargetNotInWindow'],  ###, 'Sound Right'
                #     'ax_to_plot': ['B', 'E']},
                # 3: {'Trial Start': ['No Level Cue', 'Noiseless', 'Sound Left', 'TargetNotInWindow'],  ###, 'Sound Right'
                #     'ax_to_plot': ['C', 'F']},
                # 4: {'Trial Start': ['No Level Cue', 'Noiseful', 'Sound Left', 'TargetNotInWindow'],  ###, 'Sound Right'
                #     'ax_to_plot': ['G', 'J']},
                # 3: {'Trial Start': ['No Level Cue', 'Sound Right'],
                #     'ax_to_plot': ['C', 'F']},
                #
                # 4: {'Target Time': ['No Level Cue', 'Target trials'],
                #     'ax_to_plot': ['G', 'J']},
                # 5: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                #     'ax_to_plot': ['H', 'K']},
                # 6: {'Target Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                #     'ax_to_plot': ['I', 'L']},
                #
                # 7: {'Release Time': ['No Level Cue', 'Target trials'],
                #     'ax_to_plot': ['M', 'P']},
                # 8: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                #     'ax_to_plot': ['N', 'Q']},
                # 9: {'Release Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                #     'ax_to_plot': ['O', 'R']},
            }
            # cluster_ids = [st.annotations['cluster_id'] for st in self.blocks[0].segments[0].spiketrains if st.annotations['group'] != 'noise']
            # cluster_ids = units
            windowToCheck = [-1, 2.5]  # when yu have a target/response time filter on, it's important to pay attention to this

            self.InfoOverTimeAnalysis(units, events_args, window=windowToCheck)

        # saveDir = Path(savedir)
        # with PdfPages(saveDir / f'{title}.pdf') as pdf:
        #     for clus in tqdm(units): # So this iterates through each cluster of the ones that you either called "good or multiunit" or all of the cells if you flagged that option.
        #         fig = self.testFun_unit_summary_figure(clus, events_args)
        #         #pdf.savefig(fig)
                #plt.close(fig)
    def reconstructionAnalysis(self, cluster_ids, events_args, window=[-1, 2]):

        for i, (fig_num, params) in enumerate(events_args.items()):
            evtype = list(params.keys())[0]
            filters = params[evtype]
            all_filtered_bhv = self.gather_sounds_per_trial(filters,window) # has sound files per trial, all sessions.
            clustersDone = 0
            for cluster_id in cluster_ids:
                aligned_spikes_this_cluster = self.align_neuron_to_ev_for_multi_neuron_analyses(cluster_id, evtype, filters,window=window) # this gets you a (spike count by 2) array, where the first column is the trial number corresponding to the spike, and the second column is the spike time corresponding to the spike relative to the event you specified.
                if clustersDone:
                    aligned_spikes = np.append(aligned_spikes, aligned_spikes_this_cluster, axis=0)
                else:
                    aligned_spikes = aligned_spikes_this_cluster
                clustersDone += 1

            ### begin function to get the spike rates in each trial for all neurons, prepared so that I can
            ### attempt stimulus reconstruction in each trial

            analysis_fs = 200 #2000 ### trying analysis fs of 200 just to see.
            MovingSessionWindow = 5 # number specifies how many sessions to include per loop. Make inf if you want all sessions.
            MovingSessionOverlap = 1 # number that specifies allowed overlap in the above. Should be strictly <= MovingSessionWindow. Should be >= 1.
            sessionGroupsTotal = np.size(np.unique(aligned_spikes[:,3])) - MovingSessionOverlap
            for sessionGroupNum in range(sessionGroupsTotal):
                X = []
                y = []
                currentSessionSet = np.unique(aligned_spikes[:,3])[sessionGroupNum:(MovingSessionWindow+sessionGroupNum)]
                for sessionIter, sessionNum in enumerate(currentSessionSet):
                    for trialNumIter, trialNum in enumerate(np.unique(aligned_spikes[aligned_spikes[:,3]==sessionNum,0])):
                        aligned_spikes_this_trial = aligned_spikes[(aligned_spikes[:,0] == trialNum)&(aligned_spikes[:,3]==sessionNum),:]
                        spikeTrains = np.zeros((int(int(np_fs) * (window[1] - window[0])), len(cluster_ids)))  ### one train per trial, including the excluded at the moment.
                        filtedTrains = np.zeros((int(int(np_fs) * (window[1] - window[0])), len(cluster_ids)))  # make another zeros array, same size
                        gaussianKernel = sp.signal.gaussian(3 * int(np.ceil((50 * np_fs) / 1000)), std=(((25 * np_fs) / 1000)))  ### I have no real confidence that this is correct. But the attempt is a kernel that is a size of 150 ms, with a gaussian of bandwidth (+-2std) 50 ms in the middle (that is, four times the number multiplied by np_fs, which I may change (and have already changed)). I think it works... But I should find a way to verify. Way to verify: look at the kernel surounding an isolated spike. Seems like we are at the correct size! Hurrah. (That said: I may have gotten there the wrong way, because I am using a filtfilt function which means I am actually applying this convolution twice. Oh well. Still works)
                        gaussianKernel = gaussianKernel / np.sum(gaussianKernel)
                        timeFrame = np.linspace(window[0], window[1], int((int(np_fs) * (window[1] - window[0]))))

                        for cluster_id_index, cluster_id in enumerate(cluster_ids):
                            aligned_spikes_t_and_c = aligned_spikes_this_trial[aligned_spikes_this_trial[:,2] == cluster_id,1]
                            for spikeTime in aligned_spikes_t_and_c:
                                index_min = np.argmin(abs(timeFrame - spikeTime))
                                spikeTrains[index_min,cluster_id_index] += 1
                            filtedTrains[:,cluster_id_index] = cpx_sig.filtfilt(gaussianKernel,1,spikeTrains[:,cluster_id_index]).get()




                        ### get the range of stimulus you want based on the current event you are focusing on
                        ## first,get start time for specific trial
                        temp_trialSelected = all_filtered_bhv['startTrialLick'][(all_filtered_bhv.index== trialNum)]
                        sessionSelect = all_filtered_bhv['sessionNumber'][all_filtered_bhv.index== trialNum] == sessionNum
                        trialStartTime = temp_trialSelected[sessionSelect]
                        # trialStartTime = trialStartTime.to_numpy()
                        trialStartTime = trialStartTime[trialNum] ### doing this in four lines of code is obviously ridiculous... More python issues.

                        ## next, get the time corresponding to the current event.
                        temp_ev_time = all_filtered_bhv[self.evtypes[evtype]][(all_filtered_bhv.index== trialNum)]
                        temp_ev_time = temp_ev_time[sessionSelect]
                        # temp_ev_time = temp_ev_time.to_numpy()
                        temp_ev_time = temp_ev_time[trialNum]
                        ## next, grab the whole stimulus
                        wholeStim = all_filtered_bhv['stimulus'][(all_filtered_bhv.index== trialNum)]
                        wholeStim = wholeStim[sessionSelect]
                        # wholeStim = wholeStim.to_numpy()
                        wholeStim = wholeStim[trialNum]

                        relativeTimeSamples = int(round((temp_ev_time-trialStartTime)*bhv_fs))
                        relativeWindowPositions =  [relativeTimeSamples + int(round(n*bhv_fs)) for n in window]
                        stimulusThisTrial = np.zeros((int(np.round(relativeWindowPositions[1]-relativeWindowPositions[0])), 1))


                        if (relativeWindowPositions[0] < 0)&(relativeWindowPositions[1] > len(wholeStim)): ### hnestly may be worth excluding the trial in this case, but I will assume that if yoou want
                            preStimSamples = 0 - relativeWindowPositions[0]  # samples in the pre-stim period. "Includes" the first sample of the stimulus, becausepython is 0 based.
                            stimulusThisTrial[preStimSamples:(preStimSamples+len(wholeStim))] = np.expand_dims(wholeStim, 1)
                        elif relativeWindowPositions[0] < 0:
                            preStimSamples = 0-relativeWindowPositions[0] # samples in the pre-stim period. "Includes" the first sample of the stimulus, becausepython is 0 based.
                            stimulusThisTrial[preStimSamples:] = np.expand_dims(wholeStim[:relativeWindowPositions[1]], 1)
                        elif relativeWindowPositions[1] > len(wholeStim):
                            print('not implemented')
                        elif (relativeWindowPositions[0] >= len(wholeStim))|(relativeWindowPositions[1] <= 0):
                            print('you are trying to do something weird, or a weird error happened')
                        else: # what should happen mostly
                            stimulusThisTrial = np.expand_dims(wholeStim[relativeWindowPositions[0]:relativeWindowPositions[1]], axis=1)

                        ### plotting code to show the filtered trains.
                        # n = 102
                        # plt.plot(timeFrame, spikeTrains[:, n] * max(filtedTrains[:, n]))
                        # plt.plot(timeFrame, filtedTrains[:, n])
                        # ##As this was originally designed to have equal (high) sampling rates, I used to plot sound and spikes simultaneously.### plt.plot(timeFrame, (stimulusThisTrial/max(stimulusThisTrial))* max(filtedTrains[:, n]))
                        # plt.show()
                        # stop = 'wait'

                        ### I should implement an antialiasing filter here, for 16 KHz

                        butterWorthyB, butterWorthyA = butter(6,7500,fs=bhv_fs)
                        aliasFiltedStmulus = cpx_sig.filtfilt(butterWorthyB,butterWorthyA,stimulusThisTrial,axis=0).get()
                        ### Here I begin to implement the naplib stimulus reconstruction
                        testSpectrogram = nl.features.auditory_spectrogram(aliasFiltedStmulus, bhv_fs)
                        testSpectrogram = resample(testSpectrogram, int((analysis_fs)*(window[1]-window[0]))) ### resamples to the number of samples said in second argument... By subtracting windows, we get correct length.
                        resampled_filtered_spikes = resample(filtedTrains, int((analysis_fs)*(window[1]-window[0])))
                        # plt.plot(np.linspace(window[0], window[1], (int(analysis_fs) * (window[1] - window[0]))), resampled_filtered_spikes[:, 343])
                        ### if I want to reduce the freq dimensions I can do as below.
                        resample_kwargs = {'num': 32, 'axis': 1}
                        tempList = [testSpectrogram]
                        tempList = nl.array_ops.concat_apply(tempList, resample, function_kwargs=resample_kwargs)
                        testSpectrogram = tempList[0]
                        #fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
                        # plt.imshow(testSpectrogram.T, aspect='auto', origin='lower')
                        #plt.set_title('Spectrogram of stimulus (to reconstruct)')
                        #axes[1].plot(filtedTrains)
                        #axes[1].set_title('Multichannel recording to use as input\nfeatures to reconstruction model')
                        #plt.tight_layout()
                        # plt.show()

                        X.append(resampled_filtered_spikes)
                        y.append(testSpectrogram)

                #### Try the ridge model
                tmin = -0.4 ### default -0.4
                tmax = 0 # default 0
                sfreq = analysis_fs #
                n_jobs = 16
                mdl = nl.encoding.TRF(tmin=tmin, tmax=tmax, sfreq=sfreq, estimator=Ridge(),n_jobs=n_jobs)

                X_norm = nl.preprocessing.normalize(None, X)
                nanLocations = np.isnan(X_norm[0])  ### should be the same for every trial. I check it later.
                for trialNum_inX in range(len(X_norm)):
                    nanLocationsCheck = np.isnan(X_norm[trialNum_inX])  ### should be equal to nanLocations for every trial
                    if (nanLocations != nanLocationsCheck).any():
                        print('the nan distribution didn''t work like I thought')
                    X_norm[trialNum_inX] = X_norm[trialNum_inX][:,~np.any(nanLocations,0)]
                # X_train = X_norm[1:]
                X_train = X[1:]
                y_train = y[1:]
                # X_test = [X_norm[0]]
                X_test = [X[0]]
                y_test = [y[0]]
                print(np.shape(X_train))
                mdl.fit(data=None, X=X_train, y=y_train) # should be using a training set here, defined however.

                reconstructedStim = mdl.predict(data=None,X=X_test)
                corrScore = mdl.corr(data=None, X=X_test,y=y_test).mean()

                fig, axes = plt.subplots(2, 1, figsize=(12, 6))

                axes[0].imshow(y_test[0].T, aspect=3, origin='lower')
                axes[0].set_title('True stimulus')
                axes[1].imshow(reconstructedStim[0].T, aspect=3, origin='lower')
                axes[1].set_title('Reconstructed stimulus, corr={:.3f}'.format(corrScore))
                print('stop')
                plt.show()
    def InfoOverTimeAnalysis(self, cluster_ids, events_args, window=[-1, 2]):

        spikeRateMethod = 'count'
        doInfoOverTime = 1 ### seems like this doesn't properly dodge things yet.
        for i, (fig_num, params) in enumerate(events_args.items()):
            evtype = list(params.keys())[0]
            filters = params[evtype]
            all_filtered_bhv = self.gather_sounds_per_trial(filters,window) # has sound files per trial, all sessions.
            clustersDone = 0
            for cluster_id in cluster_ids:
                aligned_spikes_this_cluster = self.align_neuron_to_ev_for_multi_neuron_analyses(cluster_id, evtype, filters,window=window) # this gets you a (spike count by 2) array, where the first column is the trial number corresponding to the spike, and the second column is the spike time corresponding to the spike relative to the event you specified.
                if clustersDone:
                    aligned_spikes = np.append(aligned_spikes, aligned_spikes_this_cluster, axis=0)
                else:
                    aligned_spikes = aligned_spikes_this_cluster
                clustersDone += 1

            ### begin function to get the spike rates in each trial for all neurons, prepared so that I can
            ### attempt stimulus reconstruction in each trial
            X = []
            y = []
            #analysis_fs = 200
            binSizeSeconds = 0.1 # Defining this so that we can iterate through bins. Efficient with RAM but not compute speed; basically we
            binSizeSamples = int(int(np_fs)*binSizeSeconds)
            trialCountPerSession = np.zeros(len(np.unique(aligned_spikes[:, 3])))
            for sessionIter, sessionNum in enumerate(np.unique(aligned_spikes[:, 3])):
                trialCountPerSession[sessionIter] = len(np.unique(aligned_spikes[aligned_spikes[:, 3] == sessionNum, 0]))
            totalTrials = int(np.sum(trialCountPerSession))
            timeBinsAll = np.arange(window[0], window[1], binSizeSeconds)
            if i == 0: # should always be first loop because that's how enumerate works.
                theMeans_unsorted = np.zeros(((len(cluster_ids), len(timeBinsAll), len(events_args.items()))))
                sortingsPerAnalysis = np.zeros((len(cluster_ids), len(events_args.items()))).astype(int)  ### size clusters by the number of analyses.
            countsPerTrialPerClusterAllTime = np.zeros((totalTrials,len(cluster_ids),len(timeBinsAll)))
            AllPCsError = np.zeros(len(timeBinsAll))
            TenPCsError = np.zeros(len(timeBinsAll))

            for winIter,winMin in enumerate(timeBinsAll):
                winMax = winMin + binSizeSeconds

                countsPerTrialPerCluster = np.zeros((totalTrials,len(cluster_ids)))
                trialCountAcrossSessions = -1 ### starting at -1 so that I can start at 0 in the loop but put the counter at the top where I can easily track it.
                for sessionIter, sessionNum in enumerate(np.unique(aligned_spikes[:,3])):
                    for trialNumIter, trialNum in enumerate(np.unique(aligned_spikes[aligned_spikes[:,3]==sessionNum,0])):
                        trialCountAcrossSessions += 1
                        aligned_spikes_this_trial = aligned_spikes[(aligned_spikes[:,0] == trialNum)&(aligned_spikes[:,3]==sessionNum),:]
                        if spikeRateMethod == 'count':
                            aligned_spikes_ThisBin = aligned_spikes_this_trial[[x for x in((aligned_spikes_this_trial[:,1]  < winMax) & (aligned_spikes_this_trial[:,1]  >= winMin))],:]
                            ### so... The above should be all spikes for this trial for the window for all clusters. My first trial has few enough spikes that I am currently paranoid.
                            for thisSpike in aligned_spikes_ThisBin:
                                currentSpikeClusterIndex = cluster_ids.index(thisSpike[2])
                                countsPerTrialPerCluster[trialCountAcrossSessions, currentSpikeClusterIndex] += 1
                            #for cluster_index,cluster_id in enumerate(cluster_ids):
                            #    countsPerTrialPerCluster(trialNumIter,cluster_index) +
                            #    countsPerTrialPerCluster(np.sum([x for x in ((aligned_spikes_this_trial[:, 1] < winMax) & (aligned_spikes_this_trial[:, 1] >= winMin))])

                        elif spikeRateMethod == 'kernel':
                            print('not implemented')
                if doInfoOverTime:
                    U, s, Vt = np.linalg.svd(countsPerTrialPerCluster, full_matrices=False)
                    V = Vt.T
                    # if we use all of the PCs we can reconstruct the noisy signal perfectly
                    S = np.diag(s)
                    Mhat = np.dot(U, np.dot(S, V.T))
                    AllPCsError[winIter] = (np.mean((countsPerTrialPerCluster - Mhat) ** 2))
                    print("Using all PCs, MSE = %.6G" % (AllPCsError[winIter]))

                    # if we use only the first 20 PCs the reconstruction is less accurate
                    PCs2Keep = 10
                    Mhat2 = np.dot(U[:, :PCs2Keep], np.dot(S[:PCs2Keep, :PCs2Keep], V[:, :PCs2Keep].T))
                    TenPCsError[winIter] = (np.mean((countsPerTrialPerCluster - Mhat2) ** 2))

                    print("Using first 10 PCs, MSE = %.6G" % (TenPCsError[winIter]))
                    countsPerTrialPerClusterAllTime[:,:,winIter] = countsPerTrialPerCluster

            ### time to look at some PSTHs in various forms.
            # countsPerTrialPerClusterAllTime is in format trials, clusters, time.
            baselineFrames = timeBinsAll < 0 ### assumes that the baseline is before whatever you are centered on: not going to be correct in a lot of cases, but will be in some.
            responseFramesForSorting = (timeBinsAll > 0)&(timeBinsAll <= 0.5)
            temp = np.zeros(np.shape(countsPerTrialPerClusterAllTime))
            for iii in range(np.size(countsPerTrialPerClusterAllTime,axis=1)):
                for timebin in range(np.size(countsPerTrialPerClusterAllTime,axis=2)):
                    temp[:,iii,timebin] = countsPerTrialPerClusterAllTime[:,iii,timebin] - np.mean(countsPerTrialPerClusterAllTime[:,iii,baselineFrames],axis=1)

            zscoredPerTrialPerClusterAllTime = np.zeros(np.shape(countsPerTrialPerClusterAllTime))
            for iii in range(np.size(countsPerTrialPerClusterAllTime, axis=1)):
                for timebin in range(np.size(countsPerTrialPerClusterAllTime, axis=2)):
                    meansThisTimeBin = temp[:, iii, timebin]
                    standardDeviationsThisLoop = np.std(temp[:, iii,:],axis=1)
                    standardDeviationsThisLoop[(meansThisTimeBin==0)&(standardDeviationsThisLoop==0)] = 1 # I think this is the way to
                    zscoredPerTrialPerClusterAllTime[:, iii, timebin] = meansThisTimeBin/standardDeviationsThisLoop
                    if np.isnan(zscoredPerTrialPerClusterAllTime).any():
                        d = 1+1
            # now, start making some plots
            theMeans_unsorted[:,:,i]=np.nanmean(zscoredPerTrialPerClusterAllTime, axis=0)
            sortingsPerAnalysis[:,i] = np.argsort(np.mean(theMeans_unsorted[:,responseFramesForSorting,i],axis=1))
            theMeans = theMeans_unsorted[sortingsPerAnalysis[:,i],:,i]
            #plt.imshow(theMeans,cmap='bwr',extent=[window[0], window[1], window[0], window[1]],vmin=-abs(np.max((np.abs(np.max(theMeans)),np.abs(np.min(theMeans))))),vmax=abs(np.max((np.abs(np.max(theMeans)),np.abs(np.min(theMeans))))))
            plt.imshow(theMeans, cmap='bwr', extent=[window[0], window[1], window[0], window[1]],
                       vmin=-2,
                       vmax=2)
            plt.colorbar()
            plt.show()
            if doInfoOverTime:
                ### time for the PCA type analysis
                constructedTitle = []
                startedConstruction = 0
                for i, param in enumerate(params[evtype]):
                    if not param == 'No Level Cue':
                        if startedConstruction > 0:
                            constructedTitle += ' ' + param
                        else:
                            constructedTitle = param
                            startedConstruction = 1
                plt.title(constructedTitle)
                plt.ylim((0.4,0.8)) ### I should make this more adaptable
                plt.plot(timeBinsAll + (binSizeSeconds / 2), TenPCsError)

                plt.show()
                varOverTimeCrude = np.zeros((np.size(timeBinsAll + (binSizeSeconds / 2)), 1))
                for iii in range(np.size(timeBinsAll + (binSizeSeconds / 2))):
                    varOverTimeCrude[iii] = np.var(countsPerTrialPerClusterAllTime[:, :, iii])

                plt.plot(timeBinsAll + (binSizeSeconds / 2), varOverTimeCrude)
                plt.show()
                multiplyThem = varOverTimeCrude*TenPCsError[:,np.newaxis]
                plt.plot(timeBinsAll + (binSizeSeconds / 2), multiplyThem)
                plt.show()
                hrowueew = 1

        print('stopping point')
        plt.figure()
        fig, axs = plt.subplots(len(events_args.items()), len(events_args.items()))
        fig.add_subplot(111, frameon=False)
        plt.ylabel("Means Sorted", fontsize=14)
        plt.xlabel("Sortings", fontsize=14)
        plt.suptitle('Order Is Noiseless and Noiseful Contralateral, then Ipsilateral')
        for i, (fig_num, params_i) in enumerate(events_args.items()):
            for j, (fig_num, params_j) in enumerate(events_args.items()):
                currentMean = theMeans_unsorted[sortingsPerAnalysis[:,i],:, j]
                axs[j, i].imshow(currentMean, cmap='bwr', extent=[window[0], window[1], window[0], window[1]], vmin=-2,vmax=2)
                #tempString = params_i[evtype][1] + ' ' + params_i[evtype][2] + " " + 'Plotting ' + params_j[evtype][1] + ' ' + params_j[evtype][2]
                #axs[i,j].colorbar()
            print('stophere')
        #plt.show()
        plt.savefig('C:\\Users\\Soraya\\Jeffrey\\Figures\\tempfig.pdf')
        print('stopping point')
    def plot_waveform(self, ax, unit):
        waveform_path = self.dp.parents[0] / 'waveforms/waveforms'
        wv_file = f'waveforms_{unit.annotations["si_unit_id"]}.npy'

        wv_data = np.load(waveform_path / wv_file)
        avg_wv = np.mean(wv_data, axis=0)
        peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])
        ax.plot(avg_wv[:, peak_channel], '-', color=3 * [.2], linewidth=5)

        util.simple_xy_axes(ax)
        ax.spines['bottom'].set_visible(False)
        ax.set_ylabel(u'Amplitude (\u03bcV)')
        ax.set_xlabel('Samples')

    def plot_channel_map(self, ax, unit):
        peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])

        probe = generate_warp32_probe(radius=30)
        values = np.zeros(len(probe.device_channel_indices))
        values[int(peak_channel)] = 1
        plot_probe(probe, ax=ax, with_channel_index=False,
                   with_device_index=True, contacts_values=values)


def run_single(session_path):
    filter_trials = {'No Level Cue'}
    # ferret = 'F1903_Trifle'
    # session = 'catgt_240622_F1903_Trifle_AM_g0'
    # # session = 'catgt_180522_Trifle_PM_g0'
    # dp = neuropixelDataPath / ferret / session / f'{session[6:]}_imec0' / 'imec0_ks2'
    # session_path = Path('/media/jules/jules_SSD/data/neural_data/Neuropixels/spikesorted/')
    # session = 'catgt_190522_F1903_Trifle_AM_g0'
    # session = 'catgt_180522_Trifle_PM_g0'
    # dp = session_path / f'{session_path.name[6:]}_imec0' / 'phy_postprocessing'
    ### dp = session_path / f'{session_path.name[6:]}_imec0' / 'phy_postprocessing'
    dp = 'D:/Jeffrey/Output/spikesorted/F2302_Challah/ACx_Challah/kilosort/phy_folder'  # Don't know if this is quite right... Let's try.

    saveDir = Path('/home/jules/Dropbox/Jules/Figures/Neuropixel/single_session_analysis')
    saveDir.mkdir(parents=False, exist_ok=True)

    dataset = NeuralDataset(dp, datatype='neuropixel')
    dataset.load()

    dataset.create_summary_pdf(saveDir, title=f'summary_data_firing_rate_{session_path.name}')


def run_concatenated():
    filter_trials = {'No Level Cue'}

    # dp = Path('/mnt/b/WarpData/behaving/output_multirec')
    # neural_data = Path('/mnt/b/WarpData/behaving/raw')
    # datatype = 'warp'

    ferret = 'F2302_Challah'

    neural_data = neuropixelDataPath#Path('Z:/Data/Neuropixels/F2302_Challah')
    #dp = Path('/mnt/a/NeuropixelData/si_neuropixel_spikesorting/F1903_Trifle/Trifle_S12_bot_NP24')
    analysisName = 'ACx_Challah' ### not intended to be permanent... But in any case, it will name the output pdf.
   # dp = Path('D:/Jeffrey/Output/spikesorted/F2302_Challah/ACx_Challah/KiloSortSortingExtractor/phy_folder')  # Don't know if this is quite right... Let's try.
    dp = Path('D:/Jeffrey/ShamDataHierarchy/F2302_Challah/ACx_Challah/All_ACx_Top/weekOf20240513/KiloSortSortingExtractor/phy_folder')

    datatype = 'neuropixels'

    # saveDir = Path('/home/jules/Dropbox/Jules/Figures/warp/concatenated_data/spikesorted_2023')
    saveDir = Path('D:/Jeffrey/Output/figures/ConcatOut')
    saveDir.mkdir(parents=False, exist_ok=True)

    dataset = concatenatedAnalysis(dp,
                                   currNeuralDataPath=neural_data,
                                   datatype=datatype,
                                   overwrite_pkl=True,
                                   )

    dataset.load()
    dataset.create_summary_pdf(saveDir, title=analysisName,doUnsorted=False)

def run_concatenated_testFunction():
    filter_trials = {'No Level Cue'}


    ferret = 'F2302_Challah'

    neural_data = neuropixelDataPath#Path('Z:/Data/Neuropixels/F2302_Challah')
    #dp = Path('/mnt/a/NeuropixelData/si_neuropixel_spikesorting/F1903_Trifle/Trifle_S12_bot_NP24')
    analysisName = 'ACx_Challah' ### not intended to be permanent... But in any case, it will name the output pdf.
    #dp = Path('D:/Jeffrey/Output/spikesorted/F2302_Challah/ACx_Challah/The_First_Day/KiloSortSortingExtractor/phy_folder')  # Don't know if this is quite right... Let's try.
    dp = Path('D:/Jeffrey/ShamDataHierarchy/F2302_Challah/ACx_Challah/All_ACx_Top/weekOf20240513/KiloSortSortingExtractor/phy_folder')
    #dp = Path('D:/Jeffrey/Output/Tones/spikesorted/F2302_Challah/06122024_AM_Challah_g0/KiloSortSortingExtractor/KiloSortSortingExtractor/phy_folder/')
    datatype = 'neuropixels'
    picklePath = Path('D:/Jeffrey/pickleFolder')
    pickleName = 'tempsave.pkl'
    loadPickle = 1
    savePickle = 0
    saveDir = Path('D:/Jeffrey/Output/figures/ConcatOut')
    saveDir.mkdir(parents=False, exist_ok=True)
    if loadPickle:
        with open(picklePath / pickleName, 'rb') as file:
            dataset = pickle.load(file)
    else:
        # saveDir = Path('/home/jules/Dropbox/Jules/Figures/warp/concatenated_data/spikesorted_2023')


        dataset = concatenatedAnalysis(dp,
                                       currNeuralDataPath=neural_data,
                                       datatype=datatype,
                                       overwrite_pkl=True,
                                       )
        dataset.load()

    ### test function stuff
    testFuncActive = True # specifies whether you use the "clustersToTest" along with anything else you write in.
    testFuncVar = {'clustersToTest': 'All','trialsToTest': 0} ### 343 good cluster in my initial debugging. Comment certainly irrelevant later.
    if savePickle:
        with open(picklePath / pickleName, 'wb') as file:
            pickle.dump(dataset, file)
    dataset.create_summary_pdf(saveDir, title=analysisName,doUnsorted=False,testFuncActive=testFuncActive
                               ,testFuncVar=testFuncVar
                               )





def main():
    data_path = Path('/media/jules/jules_SSD/data/neural_data/Neuropixels/spikesorted/')
    ferret = 'F2003_Orecchiette'
    # ferret = 'F2103_Fettucini'
    sessions = [sess for sess in data_path.glob(f'{ferret}/catgt_*')]
    for session_path in sessions:
        print(f'Summary for {session_path.name}')
        # try:
        run_single(session_path)
        # except:
        #     pass


if __name__ == '__main__':
    # main()
    #run_concatenated() # When you run the script you run run_concatenated!
    run_concatenated_testFunction() ### the idea here will be to test plotting analysis on specific neurons that I can input as arguments (in terms of cluster number). The analyses themselves will maybe vary with time; I should ideally move them into run_concatenated() or a similar function as they become standardized.
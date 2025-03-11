from pathlib import Path
from dataclasses import dataclass
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import spikeinterface.full as si
from tqdm import tqdm

from neuropixels_visualisation.helpers.util import simple_xy_axes, set_font_axes
from neuropixels_visualisation.config import neuropixelDataPath
from ..io.phynpxlio import PhyNpxlIO
from ..io.phywarpio import PhyWarpIO
from instruments.io.phyconcatrecio import PhyConcatRecIO

from instruments.helpers.extract_helpers import load_bhv_data


@dataclass
class concatenatedNeuralData:  # Reportedly this is the class Jules uses to readily link the neural data with behavior.
    ### In "NeuralDataset" the only arguments are dp and datatype.

    dp: str # path to spikesorted wrap data in phy format (probably, it is this in the non-concatenated object)
    currNeuralDataPath: str = neuropixelDataPath # If you ever use warps you can import these.
    datatype: str = 'warp'  # Either 'neuropixels' or 'warp'
    overwrite_pkl: bool = False  # overwrite something?
    sync_around_pulses: bool = True  # this is the thing where we use the aux input to sync
    window_around_pulses: float = -2

    def load(self):
        assert self.datatype in ['neuropixels', 'warp'], 'Unknown datatype'
        print("Loading data from:", self.dp)
        phy_folder = Path(self.dp) # in NeuralData, this is just self.dp
        self.evtypes = {'Trial Start': 'startTrialLick', 'Target Time': 'absoluteTargTimes',
                        'Release Time': 'absoluteRealLickRelease'}  # types of events that will be linked.

        if (phy_folder / 'blocks.pkl').exists() and not self.overwrite_pkl: # Looks like this will load old files if you don't ask it to overwrite them.
            with open(phy_folder / 'blocks.pkl', 'rb') as f:
                self.blocks = pickle.load(f)
        else:
            self.reader = PhyConcatRecIO(dirname=phy_folder,
                                         currNeuralDataPath=self.currNeuralDataPath,
                                         datatype=self.datatype,
                                         sync_around_pulses=self.sync_around_pulses,
                                         window_around_pulses=self.window_around_pulses)
            self.blocks = self.reader.read()

            for seg in self.blocks[0].segments:
                if seg.annotations['bhv_file'] is not None:
                    seg.df_bhv = load_bhv_data(
                        seg.annotations['bhv_file'])

                else:
                    seg.df_bhv = None

            #with open(phy_folder / 'blocks.pkl', 'wb') as f: ### commenting out the pickle stuff so I can bypass it.
            #    print('save pickle blocks file in:', phy_folder / 'blocks.pkl')
            #    pickle.dump(self.blocks, f)

    def get_cluster_ids(self, blocks, group=['good', 'mua']):
        if isinstance(group, list):
            cluster_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains
                           if st.annotations['group'] in group]
            return cluster_ids

        cluster_ids = [st.annotations['cluster_id'] for st in blocks[0].segments[0].spiketrains
                       if st.annotations['group'] == group]
        return cluster_ids


@dataclass
class NeuralDataset:
    # dp = path to spikesorted wrap data in phy format
    dp: str
    datatype: str # Either 'neuropixel' or 'warp'

    def load(self):
        
        assert self.datatype in ['neuropixel','warp'], 'Unknown datatype'

        self.dp = Path(self.dp)

        self.evtypes = {'Trial Start': 'startTrialLick', 'Target Time': 'absoluteTargTimes',
            'Release Time': 'absoluteRealLickRelease'} #hmm... Important context. These are the "ev" to which things will be aligned later.
            ### more notes on above... I need to add an ev type for splitting specific tokens. Or maybe I don't want to use "ev type" for that...
        if self.datatype =='neuropixel':
            phy_folder = self.dp
            self.reader = PhyNpxlIO(dirname=phy_folder) ### seems like this might read the phy file?
        elif self.datatype == 'warp':
            phy_folder = self.dp / 'phy'
            self.reader = PhyWarpIO(dirname = phy_folder)

        self.blocks = self.reader.read()
        self.df_bhv = self.reader.load_bhv_data(self.blocks[0].segments[0].annotations['bhv_file'])
        self.blocks[0].segments[0].df_bhv = self.df_bhv
        self.seg = self.blocks[0].segments[0]

        # self.spikes_array = np.concatenate((self.reader._spike_clusters, self.reader._spike_times/self.seg.annotations['sampling_frequency']), axis=1, dtype=object)
        # self.spikes_df = pd.DataFrame({'cluster': self.reader._spike_clusters, 'spike_time': self.reader._spike_times/self.seg.annotations['sampling_frequency']}, index=np.arange(self.reader._spike_times.size))

        # self.quality_metrics = pd.DataFrame(self.reader._quality_metrics)
        # self.quality_metrics.cluster_id = self.quality_metrics.cluster_id.astype('int')
        # if self.datatype == 'warp':
        #     self.quality_metrics.cluster_id = self.quality_metrics.cluster_id.astype('int') - 1


    def get_rec_objet(self): # seems like this reads the spikeglx information based on the recording path which is stored in the object.
        if self.dataype == 'neuropixel':
            rec_path = self.dp.parents[2]
            recording = si.read_spikeglx(rec_path, stream_id='imec0.ap')
            recording.annotate(is_filtered=True)

        else:
            raise NotImplementedError

        self.recording = recording
        return recording

    def get_sorting_object(self): # gets phy output I guess.
            sorting = si.read_phy(self.dp)
            self.sorting = sorting
            return sorting

 
    def align_neuron_to_ev(self, 
            cluster_id, 
            evtype,
            filter_trials = {},
            window=[-1,2],
            # fr_threshold = 1
            ): # "ev" probably means the sound signal or something.

        aligned_spikes = []
        for seg in self.blocks[0].segments:
            unit = [st for st in seg.spiketrains if st.annotations['cluster_id'] == cluster_id][0]

            # Only keep unit for this session if firing rate > 0.5/s
            # if statistics.mean_firing_rate(unit) < 0.5:
            #     continue

            filtered_bhv = seg.df_bhv.copy()
            if len(filter_trials) > 0:
                for filter in filter_trials:
                    filtered_bhv = apply_filter(filtered_bhv, filter)

            # Get event times
            ev_times = filtered_bhv[self.evtypes[evtype]].to_numpy()

            # Get spike times
            seg_aligned_spikes = align_times(unit.times.magnitude, ev_times, window)
            if len(aligned_spikes) == 0:
                aligned_spikes = seg_aligned_spikes
            else:
                seg_aligned_spikes[:,0] = seg_aligned_spikes[:,0] + aligned_spikes[-1,0]
                aligned_spikes = np.concatenate((aligned_spikes, seg_aligned_spikes))

        return aligned_spikes

    def create_summary_pdf(self,
            saveDir,
            title='summary_pdf',
            window=[-0.5, 1],
            binsize = 0.02): # a plotting function...
            
        block = self.blocks[0]

        units = [st.annotations['cluster_id'] for st in block.segments[0].spiketrains] # if st.annotations['group'] == 'good']
        self._summary_pdf(units, title, saveDir, window=window, binsize=binsize)


    def _summary_pdf(self, 
                    units, 
                    title, 
                    savedir,
                    window=[-0.5, 1],
                    binsize = 0.02):

        events_args = {1:{'Trial Start': ['No Level Cue'],
                        'ax_to_plot': ['A','D']},

                        2:{'Trial Start': ['No Level Cue', 'Sound Left'],
                            'ax_to_plot': ['B','E']},
                        3:{'Trial Start': ['No Level Cue', 'Sound Right'],
                            'ax_to_plot': ['C','F']},
                        
                        4:{'Target Time': ['No Level Cue', 'Target trials'],
                            'ax_to_plot': ['G','J']},
                        5:{'Target Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                            'ax_to_plot': ['H','K']},
                        6:{'Target Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                            'ax_to_plot': ['I','L']},
               
                        7:{'Release Time': ['No Level Cue', 'Target trials'],
                            'ax_to_plot': ['M','P']},
                        8:{'Release Time': ['No Level Cue', 'Target trials', 'Sound Left'],
                            'ax_to_plot': ['N','Q']},
                        9:{'Release Time': ['No Level Cue', 'Target trials', 'Sound Right'],
                            'ax_to_plot': ['O','R']},}

        # cluster_ids = [st.annotations['cluster_id'] for st in self.blocks[0].segments[0].spiketrains if st.annotations['group'] != 'noise']
        # cluster_ids = units
        saveDir = Path(savedir)
        with PdfPages(saveDir / f'{title}.pdf') as pdf:
            print(f'Saving summary figures as pdf for {self.dp}')
            for clus in tqdm(units):
                fig = self._unit_summary_figure(clus, events_args, window=window, binsize=binsize)
                pdf.savefig(fig)
                plt.close(fig)


    def _unit_summary_figure(self, 
                        cluster_id, 
                        events_args,
                        window=[-0.5, 1],
                        binsize = 0.02): ### This is either just the plotting function or it also does the analysis. I'll let you know what I find out.
        
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
        """ ### Seems like this might be a way of subplotting, which excites me.

            # STT
            # STT
            
        fig = plt.figure(figsize=(20,15), dpi=300)
        ax_dict = fig.subplot_mosaic(mosaic)
        ax_keys = list(ax_dict.keys())

        bins = np.arange(window[0],window[1],binsize)

        for i, (fig_num, params) in enumerate(events_args.items()):
            evtype = list(params.keys())[0]
            filters = params[evtype]
            aligned_spikes = self.align_neuron_to_ev(cluster_id, evtype, filters, window=window)
            
            axes = [ax_dict[k] for k in params['ax_to_plot']]
            # 0, 1, 2 when i == 0, 1, 2 and 6, 7, 8 when i == 3, 4, 5

            axes[0].scatter(x = aligned_spikes[:,1], y = aligned_spikes[:,0], 
                s = 6, c = colors[evtype], alpha = 0.8, edgecolors='none'
                )
            psth, edges = np.histogram(aligned_spikes[:,1], bins)
            psthfr = (psth/len(np.unique(aligned_spikes[:,0])))/binsize
            # zscore = (psthfr - np.mean(psthfr))/np.std(psthfr)
            # zscore = (psthfr - np.mean(psthfr[:int((0-window[0])/binsize)]))/np.std(psthfr[:int((0-window[0])/binsize)])


            # axes[1].hist(aligned_spikes[:,1], bins = 100, 
            #     color = colors[evtype], alpha = 0.8)
            axes[1].plot(edges[:-1], psthfr, color = colors[evtype], alpha = 0.8, linewidth=4)
            axes[1].set_ylabel('Firing rate (spikes/s)', fontweight='bold')
            axes[1].set_ylim(bottom=0)

            for ax in axes:
                ax.axvline(0, color = colors[evtype], linestyle = '--', linewidth=2)
                ax.set_xlabel('Time (s)', fontweight='bold')
                simple_xy_axes(ax)
                set_font_axes(ax, add_size=15)


            axes[0].set_title(f'Unit {cluster_id} {evtype} {" ".join(filters)}',fontweight='bold', fontsize=12)
        
        unit = [st for st in self.blocks[0].segments[0].spiketrains if st.annotations['cluster_id'] == cluster_id][0]        
        
        # self.plot_waveform(ax_dict['S'], unit, waveform_path=self.dp / 'waveforms') # plot waveform
        # self.plot_channel_map(ax_dict['T'], unit)

        quality = unit.annotations['group']
        fig.suptitle(f'Unit {cluster_id} {quality}', fontweight='bold', fontsize=20)

        fig.tight_layout()

        return fig

    # def plot_waveform(self, ax, unit, waveform_path = None):
    #     if waveform_path is None:
    #         waveform_path = self.dp.parents[0] / 'waveforms/waveforms'
    #     wv_file = f'waveforms_{unit.annotations["si_unit_id"]}.npy'

    #     wv_data = np.load(waveform_path / wv_file)
    #     avg_wv = np.mean(wv_data, axis=0)
    #     peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])
    #     ax.plot(avg_wv[:, peak_channel], '-', color=3*[.2], linewidth=5)

    #     util.simple_xy_axes(ax)
    #     ax.spines['bottom'].set_visible(False)
    #     ax.set_ylabel(u'Amplitude (\u03bcV)')
    #     ax.set_xlabel('Samples')


    # def plot_channel_map(self, ax, unit):
    #     peak_channel = int(unit.annotations['peak_info']['max_on_channel_id'])

    #     probe = generate_warp32_probe(radius=30)
    #     values = np.zeros(len(probe.device_channel_indices))
    #     values[int(peak_channel)] = 1
    #     plot_probe(probe, ax=ax, with_channel_index=False,
    #         with_device_index=True,contacts_values=values)

def align_times(times, events, window=[-1, 1]):
    """
    Aligns times to events.
    
    Parameters
    ----------
    times : np.array
        Spike times (in seconds).
    events : np.array
        Event times (in seconds).
    window : list
        Window around event (in seconds).

    Returns
    -------
    aligned_times : np.array
        Aligned spike times.
    """

    t = np.sort(times)
    aligned_times = np.array([])
    for i, e in enumerate(events):
        ts = t-e # ts: t shifted
        tsc = ts[(ts>=window[0])&(ts<=window[1])] # tsc: ts clipped
        al_t = np.full((tsc.size, 2), i, dtype='float')
        al_t[:,1] = tsc
        if len(aligned_times) == 0:
            aligned_times = al_t
        else:
            aligned_times = np.concatenate((aligned_times, al_t))

    return aligned_times

def apply_filter(df, filter):
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
    if filter == 'Hit Trials':
        df = apply_filter(df, 'Target trials')
        return df.loc[df.relReleaseTime > 0 & df.relReleaseTime < 2]
    if filter == 'Noise Trials':
        return df.loc[df.currNoiseAtten <= 0]
    if filter == 'Silence Trials':
        return df.loc[df.currNoiseAtten > 60]
    else:
        return f'Filter "{filter}" not found'
    

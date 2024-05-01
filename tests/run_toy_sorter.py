# %%
import spikeinterface.extractors as se
import spikeinterface.sorters as ss


test_recording, _ = se.toy_example(
    duration=30,
    seed=0,
    num_channels=64,
    num_segments=1
)
test_recording = test_recording.save(folder="test-rec", overwrite=True, verbose=True)
# %%
sorter_name = 'kilosort4'
sorting = ss.run_sorter(
    sorter_name=sorter_name,
    recording=test_recording,
    output_folder=sorter_name,
    remove_existing_folder=True,
    )

print(sorting)


# %%
import spikeinterface.core as sc
sc.load_extractor('/mnt/a/NeuropixelData/output_jeffrey/tempDir/')
# %%

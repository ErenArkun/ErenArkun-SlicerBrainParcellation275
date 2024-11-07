import os
import slicer
import vtk

script_dir = os.path.dirname(os.path.realpath(__file__))
txt_path = os.path.join(script_dir, "..", "level", "Level5.txt")

def update_segment_names(outputFolder, txt_file_path=txt_path):
    subfolders = [f for f in os.listdir(outputFolder) if os.path.isdir(os.path.join(outputFolder, f))]
    for subfolder in subfolders:
        subfolder_path = os.path.join(outputFolder, subfolder)

        segmentation_node = None
        for j in os.listdir(subfolder_path):
            if j.endswith('.nii') and "_280" in j:
                nii_segment_file_path = os.path.join(subfolder_path, j)
                segmentation_node = slicer.util.loadSegmentation(nii_segment_file_path)
                if not segmentation_node:
                    print(f"Failed to load segmentation file: {nii_segment_file_path}")
                    continue
                file_name = j
                break

        if not segmentation_node:
            print(f"Could not load segmentation node under {subfolder_path}.")
            continue

        segment_names = []
        try:
            with open(txt_file_path, 'r') as x:
                for line in x:
                    columns = line.strip().split()
                    if len(columns) >= 2:
                        segment_names.append(columns[1])
        except FileNotFoundError:
            print(f"TXT file not found: {txt_file_path}")
            continue

        segmentation = segmentation_node.GetSegmentation()
        segment_ids = vtk.vtkStringArray()
        segmentation.GetSegmentIDs(segment_ids)

        if segment_ids.GetNumberOfValues() != len(segment_names):
            print(f"Warning: {len(segment_names)} segments were expected but {segment_ids.GetNumberOfValues()} were found.")

        missing_segments = []

        for i in range(segment_ids.GetNumberOfValues()):
            segment_id = segment_ids.GetValue(i)
            segment = segmentation.GetSegment(segment_id)

            if segment is None:
                print(f"Segment ID: {segment_id} is invalid or could not be loaded.")
                missing_segments.append(segment_id)
                continue

            old_name = segment.GetName()
            new_name = segment_names[i] if i < len(segment_names) else f"Segment_{i}"
            segment.SetName(new_name)
            print(f"Old name: {old_name}, New name: {new_name}")

            display_node = segmentation_node.GetDisplayNode()
            if display_node:
                display_node.SetSegmentVisibility(segment_id, True)
            else:
                print(f"Segment ID: Could not set view for {segment_id}.")
                
            segment_name = segment.GetName()
            visibility = display_node.GetSegmentVisibility(segment_id) if display_node else "Unable to Set Visibility"
            print(f"Segment ID: {segment_id}, Segment Name: {segment_name}, Visibility Status: {visibility}")

        if missing_segments:
            print(f"{len(missing_segments)} segments are invalid: {missing_segments}")

        file_name = file_name.replace(".nii", ".seg.nrrd")
        output_segmentation_path = os.path.join(subfolder_path, file_name)
        if not slicer.util.saveNode(segmentation_node, output_segmentation_path):
            print(f"Error occurred while saving segmentation: {output_segmentation_path}")
        else:
            print(f"Segmentation saved successfully: {output_segmentation_path}")

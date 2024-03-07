### file flow
File Execution Order

1. ../sample_processing.ipynb
- setting low score threshold from inference result (confidence score)
- image processing (make output folder/image_files)

2. calculate_metric.ipynb
- from inference result
- output: processed_image.csv

```
1) predict confidence score(from original image), 2) ViTPose inference(confidence score from processed image)
```

3. result_after_processing.ipynb
- check score after image processing
- output: increased/decreased id.json

4. comparison_in_de.ipynb
- comparison measures between original image dataframe and processed image dataframe

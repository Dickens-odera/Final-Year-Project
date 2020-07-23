package in.notyouraveragedev.tensor_image_classification.classifier;

import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The class that performs image classification usign tensor flow model
 *
 *
 */
public class ImageClassifier {

    /**
     * Quantized MobileNet models requires additional dequantization to the output probability.
     */
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 255.0f;

    /**
     * The quantized model does not require normalization, thus set mean as 0.0f, and std as 1.0f to
     * bypass the normalization.
     */
    private static final float IMAGE_STD = 1.0f;
    private static final float IMAGE_MEAN = 0.0f;

    private static final int MAX_SIZE = 5;

    /**
     * Image size along the x axis.
     */
    private final int imageResizeX;
    /**
     * Image size along the y axis.
     */
    private final int imageResizeY;

    /**
     * Labels corresponding to the output of the vision model.
     */
    private final List<String> labels;

    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    private final Interpreter tensorClassifier;
    /**
     * Input image TensorBuffer.
     */
    private TensorImage inputImageBuffer;
    /**
     * Output probability TensorBuffer.
     */
    private final TensorBuffer probabilityImageBuffer;
    /**
     * Processer to apply post processing of the output probability.
     */
    private final TensorProcessor probabilityProcessor;

    /**
     * Creates a classifier
     *
     * @param activity the current activity
     * @throws IOException
     */
    public ImageClassifier(Activity activity) throws IOException {
        /*
         * The loaded TensorFlow Lite model.
         */
        MappedByteBuffer classifierModel = FileUtil.loadMappedFile(activity,
                "model.tflite");
        // Loads labels out from the label file.
        labels = FileUtil.loadLabels(activity, "maize_classifier_labels.txt");

        tensorClassifier = new Interpreter(classifierModel, null);

        // Reads type and shape of input and output tensors, respectively. [START]
        int imageTensorIndex = 0; // input
        int probabilityTensorIndex = 0;// output

        int[] inputImageShape = tensorClassifier.getInputTensor(imageTensorIndex).shape();
        DataType inputDataType = tensorClassifier.getInputTensor(imageTensorIndex).dataType();

        int[] outputImageShape = tensorClassifier.getOutputTensor(probabilityTensorIndex).shape();
        DataType outputDataType = tensorClassifier.getOutputTensor(probabilityTensorIndex).dataType();

        imageResizeX = inputImageShape[1];
        imageResizeY = inputImageShape[2];
        // Reads type and shape of input and output tensors, respectively. [END]

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(inputDataType);

        // Creates the output tensor and its processor.
        probabilityImageBuffer = TensorBuffer.createFixedSize(outputImageShape, outputDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD))
                .build();
    }

    /**
     * method runs the inference and returns the classification results
     *
     * @param bitmap            the bitmap of the image
     * @param sensorOrientation orientation of the camera
     * @return classification results
     */
    public List<Recognition> recognizeImage(final Bitmap bitmap, final int sensorOrientation) {
        List<Recognition> recognitions = new ArrayList<>();

        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        tensorClassifier.run(inputImageBuffer.getBuffer(), probabilityImageBuffer.getBuffer().rewind());

        // Gets the map of label and probability.
        Map<String, Float> labelledProbability = new TensorLabel(labels,
                probabilityProcessor.process(probabilityImageBuffer)).getMapWithFloatValue();

        for (Map.Entry<String, Float> entry : labelledProbability.entrySet()) {
            recognitions.add(new Recognition(entry.getKey(), entry.getValue()));
        }

        // Find the best classifications by sorting predicitons based on confidence
        Collections.sort(recognitions);
        // returning top 5 predicitons
        return recognitions.subList(0, MAX_SIZE);
    }

    /**
     * loads the image into tensor input buffer and apply pre processing steps
     *
     * @param bitmap            the bit map to be loaded
     * @param sensorOrientation the sensor orientation
     * @return the image loaded tensor input buffer
     */
    private TensorImage loadImage(Bitmap bitmap, int sensorOrientation) {
        // Loads bitmap into a TensorImage.
        inputImageBuffer.load(bitmap);

        int noOfRotations = sensorOrientation / 90;
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

        // Creates processor for the TensorImage.
        // pre processing steps are applied here
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageResizeX, imageResizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(new Rot90Op(noOfRotations))
                .add(new NormalizeOp(IMAGE_MEAN, IMAGE_STD))
                .build();
        return imageProcessor.process(inputImageBuffer);
//        return inputImageBuffer;
    }

    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    public class Recognition implements Comparable {
        /**
         * Display name for the recognition.
         */
        private String name;
        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private float confidence;

        public Recognition() {
        }

        public Recognition(String name, float confidence) {
            this.name = name;
            this.confidence = confidence;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public float getConfidence() {
            return confidence;
        }

        public void setConfidence(float confidence) {
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return "Recognition{" +
                    "name='" + name + '\'' +
                    ", confidence=" + confidence +
                    '}';
        }

        @Override
        public int compareTo(Object o) {
            return Float.compare(((Recognition) o).confidence, this.confidence);
        }
    }
}

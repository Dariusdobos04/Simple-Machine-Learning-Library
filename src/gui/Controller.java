package gui;

import classifiers.KNNModel;
import classifiers.NaiveBayes;
import classifiers.Perceptron;
import data.DataLoader;
import data.DataSplitter;
import data.Instance;
import evaluation.*;
import javafx.beans.property.ReadOnlyStringWrapper;
import javafx.collections.FXCollections;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;
import javafx.scene.control.Alert;
import utils.EuclideanDistance;
import utils.Triple;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Controller {

    @FXML
    private ComboBox<String> modelComboBox;

    @FXML
    private TextField parameterTextField;

    @FXML
    private Button loadDataButton;

    @FXML
    private Button trainButton;

    @FXML
    private TextArea outputTextArea;

    @FXML
    private Button browseButton;

    @FXML
    private TextField filePathTextField;

    @FXML
    private TextField trainPercentTextField;

    @FXML
    private TextField validationPercentTextField;

    @FXML
    private TextField testPercentTextField;

    @FXML
    private TableView<List<String>> confusionMatrixTable;

    @FXML
    private TableColumn<List<String>, String> classColumn;


    private List<Instance<Double, Integer>> data;

    private final Map<String, Double> accuracyMap = new HashMap<>();
    private final Map<String, Double> precisionMap = new HashMap<>();
    private final Map<String, Double> recallMap = new HashMap<>();
    private final Map<String, Double> f1ScoreMap = new HashMap<>();

    @FXML
    public void initialize() {
        modelComboBox.setItems(FXCollections.observableArrayList("KNN", "Naive Bayes", "Perceptron"));

        modelComboBox.setOnAction(event -> {
            String selectedModel = modelComboBox.getValue();
            switch (selectedModel) {
                case "KNN":
                    parameterTextField.clear();
                    parameterTextField.setDisable(false);
                    parameterTextField.setPromptText("k (e.g., 3)");
                    break;
                case "Perceptron":
                    parameterTextField.clear();
                    parameterTextField.setDisable(false);
                    parameterTextField.setPromptText("Learning Rate (e.g., 0.01)");
                    break;
                case "Naive Bayes":
                    parameterTextField.clear();
                    parameterTextField.setDisable(true);
                    parameterTextField.setPromptText("");
                    break;
                default:
                    parameterTextField.clear();
                    parameterTextField.setDisable(true);
                    parameterTextField.setPromptText("");
                    break;
            }
        });

        classColumn.setCellValueFactory(param -> null);
    }

    @FXML
    private void browseFile() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select CSV Data File");
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File selectedFile = fileChooser.showOpenDialog(browseButton.getScene().getWindow());
        if (selectedFile != null) {
            filePathTextField.setText(selectedFile.getAbsolutePath());
        }
    }

    @FXML
    private void loadData() {
        outputTextArea.clear();
        String filePath = filePathTextField.getText().trim();
        if (filePath.isEmpty()) {
            showErrorAlert("Error", "Please choose a file first.");
            return;
        }

        try {
            DataLoader<Double, Integer> dataLoader = new DataLoader<>();
            data = dataLoader.loadFromCSV(filePath, tokens -> {
                List<Double> features = Arrays.stream(tokens)
                        .limit(tokens.length - 1)
                        .map(Double::parseDouble)
                        .collect(Collectors.toList());
                Integer label = Integer.parseInt(tokens[tokens.length - 1]);
                return new Instance<>(features, label);
            });
            showInformationAlert("Success", "Data loaded successfully from " + filePath + ".");
        } catch (IOException e) {
            showErrorAlert("Error", "Unable to load file: " + e.getMessage());
        } catch (NumberFormatException e) {
            showErrorAlert("Error", "Invalid data format in file.");
        }
    }

    @FXML
    private void trainModel() {
        outputTextArea.clear();

        String selectedModel = modelComboBox.getValue();
        if (selectedModel == null) {
            showErrorAlert("Error", "Please select a model.");
            return;
        }
        if (data == null) {
            showErrorAlert("Error", "Please load data first.");
            return;
        }

        double trainPercent, validationPercent, testPercent;
        try {
            trainPercent = Double.parseDouble(trainPercentTextField.getText().trim()) / 100.0;
            validationPercent = Double.parseDouble(validationPercentTextField.getText().trim()) / 100.0;
            testPercent = Double.parseDouble(testPercentTextField.getText().trim()) / 100.0;

            if (trainPercent + validationPercent + testPercent != 1.0) {
                showErrorAlert("Error", "Train/Validation/Test percentages must sum up to 100%.");
                return;
            }
        } catch (NumberFormatException e) {
            showErrorAlert("Error", "Please enter valid percentages.");
            return;
        }

        try {
            DataSplitter<Double, Integer> splitter = new DataSplitter<>();
            Triple<List<Instance<Double, Integer>>, List<Instance<Double, Integer>>, List<Instance<Double, Integer>>> splitData =
                    splitter.split(data, trainPercent, validationPercent);
            List<Instance<Double, Integer>> trainingData = splitData.getFirst();
            List<Instance<Double, Integer>> testData = splitData.getSecond();

            List<Integer> predictions;

            switch (selectedModel) {
                case "KNN" -> {
                    int k;
                    try {
                        k = Integer.parseInt(parameterTextField.getText().trim());
                    } catch (NumberFormatException e) {
                        showErrorAlert("Error", "Please enter a valid integer for k.");
                        return;
                    }
                    KNNModel<Double, Integer> knnModel = new KNNModel<>(k, new EuclideanDistance());
                    knnModel.train(trainingData);
                    predictions = knnModel.test(testData);
                }
                case "Naive Bayes" -> {
                    NaiveBayes<Double, Integer> nbModel = new NaiveBayes<>();
                    nbModel.train(trainingData);
                    predictions = nbModel.test(testData);
                }
                case "Perceptron" -> {
                    double learningRate;
                    try {
                        learningRate = Double.parseDouble(parameterTextField.getText().trim());
                    } catch (NumberFormatException e) {
                        showErrorAlert("Error", "Please enter a valid number for learning rate.");
                        return;
                    }
                    int epochs = 1000;
                    Perceptron<Double, Integer> perceptron = new Perceptron<>(learningRate, epochs);
                    perceptron.train(trainingData);
                    predictions = perceptron.test(testData);
                }
                default -> {
                    showErrorAlert("Error", "Unknown model selected.");
                    return;
                }
            }

            if (predictions != null) {
                evaluatePerformance(testData, predictions, selectedModel);
            }

        } catch (Exception e) {
            showErrorAlert("Error", "Error training model: " + e.getMessage());
        }
    }

    private void evaluatePerformance(List<Instance<Double, Integer>> testData, List<Integer> predictions, String modelName) {
        try {
            outputTextArea.clear();

            List<Integer> trueLabels = testData.stream().map(Instance::getLabel).collect(Collectors.toList());
            int positiveLabel = 1;

            EvaluationMeasure<Integer> accuracyMeasure = new Accuracy<>();
            EvaluationMeasure<Integer> precisionMeasure = new Precision<>(positiveLabel);
            EvaluationMeasure<Integer> recallMeasure = new Recall<>(positiveLabel);
            EvaluationMeasure<Integer> f1ScoreMeasure = new F1Score<>();

            double accuracy = accuracyMeasure.evaluate(trueLabels, predictions);
            double precision = precisionMeasure.evaluate(trueLabels, predictions);
            double recall = recallMeasure.evaluate(trueLabels, predictions);
            double f1Score = f1ScoreMeasure.evaluate(trueLabels, predictions);

            outputTextArea.appendText("Evaluation Results for " + modelName + ":\n");
            outputTextArea.appendText(String.format("Accuracy: %.4f\n", accuracy));
            outputTextArea.appendText(String.format("Precision: %.4f\n", precision));
            outputTextArea.appendText(String.format("Recall: %.4f\n", recall));
            outputTextArea.appendText(String.format("F1 Score: %.4f\n", f1Score));

            accuracyMap.put(modelName, accuracy);
            precisionMap.put(modelName, precision);
            recallMap.put(modelName, recall);
            f1ScoreMap.put(modelName, f1Score);

            displayConfusionMatrix(trueLabels, predictions);
        } catch (Exception e) {
            showErrorAlert("Error", "Error evaluating model: " + e.getMessage());
        }
    }

    private void displayConfusionMatrix(List<Integer> trueLabels, List<Integer> predictions) {
        Set<Integer> uniqueClasses = new HashSet<>(trueLabels);
        uniqueClasses.addAll(predictions);
        List<Integer> sortedClasses = uniqueClasses.stream().sorted().toList();

        int size = sortedClasses.size();
        int[][] matrix = new int[size][size];

        Map<Integer, Integer> classIndexMap = new HashMap<>();
        for (int i = 0; i < size; i++) {
            classIndexMap.put(sortedClasses.get(i), i);
        }

        for (int i = 0; i < trueLabels.size(); i++) {
            int actual = trueLabels.get(i);
            int predicted = predictions.get(i);
            matrix[classIndexMap.get(actual)][classIndexMap.get(predicted)]++;
        }

        confusionMatrixTable.getColumns().clear();

        TableColumn<List<String>, String> classLabelColumn = new TableColumn<>("Class");
        classLabelColumn.setCellValueFactory(param ->
                new ReadOnlyStringWrapper(param.getValue().get(0))
        );
        confusionMatrixTable.getColumns().add(classLabelColumn);

        for (int colIndex = 0; colIndex < size; colIndex++) {
            Integer predictedClass = sortedClasses.get(colIndex);
            TableColumn<List<String>, String> col = new TableColumn<>("Pred=" + predictedClass);
            final int colIdx = colIndex + 1;
            col.setCellValueFactory(param ->
                    new ReadOnlyStringWrapper(param.getValue().get(colIdx))
            );
            confusionMatrixTable.getColumns().add(col);
        }

        List<List<String>> rows = new ArrayList<>();
        for (int rowIndex = 0; rowIndex < size; rowIndex++) {
            List<String> row = new ArrayList<>();
            row.add("Actual=" + sortedClasses.get(rowIndex));
            for (int colIndex = 0; colIndex < size; colIndex++) {
                row.add(String.valueOf(matrix[rowIndex][colIndex]));
            }
            rows.add(row);
        }

        confusionMatrixTable.setItems(FXCollections.observableArrayList(rows));
    }

    private void showErrorAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void showInformationAlert(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}

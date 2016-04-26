package az.neurogh.demo;

import java.util.Arrays;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.IterativeLearning;
import org.neuroph.nnet.learning.LMS;

public class TestBean implements LearningEventListener {

	public void testSimplePerceptron_And(){
		DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        
        SimplePerceptron myPerceptron = new SimplePerceptron(2);

        myPerceptron.getLearningRule().addListener(this);
        
        System.out.println("Training neural network SimplePerceptron_And...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network SimplePerceptron_And");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
	public void testSimplePerceptron_Or(){
		DataSet trainingSet = new DataSet(2, 1);
        trainingSet.addRow(new DataSetRow(new double[]{0, 0}, new double[]{0}));
        trainingSet.addRow(new DataSetRow(new double[]{0, 1}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 0}, new double[]{1}));
        trainingSet.addRow(new DataSetRow(new double[]{1, 1}, new double[]{1}));
        
        SimplePerceptron myPerceptron = new SimplePerceptron(2);

        myPerceptron.getLearningRule().addListener(this);
        
        System.out.println("Training neural network SimplePerceptron_Or...");
        myPerceptron.learn(trainingSet);
        
        System.out.println("Testing trained neural network SimplePerceptron_Or");
        testNeuralNetwork(myPerceptron, trainingSet);
	}
	
	public static void testNeuralNetwork(NeuralNetwork neuralNet, DataSet testSet) {

        for(DataSetRow testSetRow : testSet.getRows()) {
            neuralNet.setInput(testSetRow.getInput());
            neuralNet.calculate();
            double[] networkOutput = neuralNet.getOutput();

            System.out.print("Input: " + Arrays.toString( testSetRow.getInput() ) );
            System.out.println("Output: " + Arrays.toString( networkOutput) );
        }
    }
	
	@Override
	public void handleLearningEvent(LearningEvent event) {
		IterativeLearning bp = (IterativeLearning)event.getSource();
        System.out.println("iterate:"+bp.getCurrentIteration()); 
        System.out.println(Arrays.toString(bp.getNeuralNetwork().getWeights()));
		
	}

}
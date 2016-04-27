package az.neurogh.demo;

import java.util.ArrayList;
import java.util.List;

import org.neuroph.core.Connection;
import org.neuroph.core.Layer;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.Neuron;
import org.neuroph.core.Weight;
import org.neuroph.core.input.And;
import org.neuroph.core.learning.SupervisedLearning;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

public class MultiPerceptron_AndOutputNoLearn extends NeuralNetwork<SupervisedLearning>{

	private static final long serialVersionUID = 6340802492716527435L;
	
	public MultiPerceptron_AndOutputNoLearn(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
		NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("transferFunction", transferFunctionType);
        List<Integer> neuronsInLayersVector = new ArrayList<>();
        for (int i = 0; i < neuronsInLayers.length; i++) {
            neuronsInLayersVector.add(new Integer(neuronsInLayers[i]));
        }
        this.createNetwork(neuronsInLayersVector, neuronProperties);
    }
	
	private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
		
		this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);
		
		NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
		
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);
        
        Layer prevLayer = layer;
        int layerIdx = 1;
        for (layerIdx = 1; layerIdx < neuronsInLayers.size()-1; layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);
            layer.addNeuron(new BiasNeuron());
            this.addLayer(layer);
            if (prevLayer != null) {
                ConnectionFactory.fullConnect(prevLayer, layer);
            }
            prevLayer = layer;
        }
        
        //Below - AndOutputNoLearn -- hard code weight, here only 1 hidden layer.
        
        Neuron n1=layer.getNeuronAt(0);
        Connection[] c1=n1.getInputConnections();
        c1[0].setWeight(new Weight(2));
        c1[1].setWeight(new Weight(2));
        c1[2].setWeight(new Weight(-1));
        
        Neuron n2=layer.getNeuronAt(1);
        Connection[] c2=n2.getInputConnections();
        c2[0].setWeight(new Weight(-2));
        c2[1].setWeight(new Weight(-2));
        c2[2].setWeight(new Weight(3));
        
        Integer neuronsNum = neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        
        outProperties.put("inputFunction", And.class);
        
        layer = LayerFactory.createLayer(neuronsNum, outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer, layer);
        prevLayer = layer;
       
        NeuralNetworkFactory.setDefaultIO(this);
		
	}

}

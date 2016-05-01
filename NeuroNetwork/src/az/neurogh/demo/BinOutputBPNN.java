package az.neurogh.demo;

import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Linear;
import org.neuroph.core.transfer.Step;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.random.NguyenWidrowRandomizer;

public class BinOutputBPNN  extends BPNeuroNetwork{
	
	private static final long serialVersionUID = 4576224434781327843L;

	public BinOutputBPNN(TransferFunctionType sigmoid, int... neuronsInLayers) {
		super(sigmoid,neuronsInLayers);
	}

	private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
    	this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);
    	
    	NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);
        layer.addNeuron(new BiasNeuron());
        this.addLayer(layer);
        
        Layer prevLayer = layer;
        int layerIdx = 1;
        for (layerIdx = 1; layerIdx < neuronsInLayers.size(); layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);
            this.addLayer(layer);
            
            if(layerIdx != neuronsInLayers.size()-1){
            	layer.addNeuron(new BiasNeuron());
            }
            
            if (prevLayer != null) {
                ConnectionFactory.fullConnect(prevLayer, layer);
            }

            prevLayer = layer;
        }
        
        Integer neuronsNum = neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        
        outProperties.put("transferFunction", Step.class);
        layer = LayerFactory.createLayer(neuronsNum, outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer, layer);
        prevLayer = layer;
        
        NeuralNetworkFactory.setDefaultIO(this);
        
        this.setLearningRule(new BackPropagation());
        
        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));
    }

}

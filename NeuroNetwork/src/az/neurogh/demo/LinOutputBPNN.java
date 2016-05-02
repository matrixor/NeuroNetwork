package az.neurogh.demo;

import java.util.List;

import org.neuroph.core.Layer;
import org.neuroph.core.transfer.Linear;
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

public class LinOutputBPNN extends BPNeuroNetwork{
	private static final long serialVersionUID = -845079443629155109L;
	public LinOutputBPNN(int... neuronsInLayers) {
        super(neuronsInLayers);
    }

	public LinOutputBPNN(TransferFunctionType sigmoid, int... neuronsInLayers) {
		super(sigmoid,neuronsInLayers);
	}
	
	protected void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
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

        Integer neuronsNum = neuronsInLayers.get(layerIdx);
        NeuronProperties outProperties=new NeuronProperties();
        outProperties.put("transferFunction", Linear.class);
        layer = LayerFactory.createLayer(neuronsNum, outProperties);
        this.addLayer(layer);
        ConnectionFactory.fullConnect(prevLayer, layer);
        prevLayer = layer;

        NeuralNetworkFactory.setDefaultIO(this);

        //this.setLearningRule(new SigmoidDeltaRule());
        this.setLearningRule(new BackPropagation());

        this.randomizeWeights(new NguyenWidrowRandomizer(-0.7, 0.7));
    }
}

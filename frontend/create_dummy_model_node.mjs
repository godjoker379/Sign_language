import * as tf from '@tensorflow/tfjs-node';

async function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 32, activation: 'relu', inputShape: [63] }));
  model.add(tf.layers.dense({ units: 16, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 3, activation: 'softmax' }));
  
  model.compile({ optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  
  await model.save('file://./public/model');
  console.log('Dummy model created successfully in public/model');
}

createModel();

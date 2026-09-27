import React, { useRef, useEffect, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<string>('');

  useEffect(() => {
    // Load the model
    const loadModel = async () => {
      try {
        // We will host the model in the public folder
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        setModel(loadedModel);
        console.log('Model loaded');
      } catch (e) {
        console.error('Failed to load model', e);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    let handLandmarker: HandLandmarker;
    let animationFrameId: number;

    const initializeMediaPipe = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
          delegate: 'GPU'
        },
        runningMode: 'VIDEO',
        numHands: 1
      });

      startCamera();
    };

    const startCamera = async () => {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener('loadeddata', predictWebcam);
        }
      }
    };

    const predictWebcam = async () => {
      if (!videoRef.current || !canvasRef.current || !handLandmarker) return;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');

      if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, performance.now());
        
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
        
        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0];
          // Draw landmarks
          drawLandmarks(ctx, landmarks, canvas.width, canvas.height);
          
          if (model) {
            // Process landmarks for the model
            const inputData = landmarks.map(lm => [lm.x, lm.y, lm.z]).flat();
            // Optional: normalize inputData
            
            const tensor = tf.tensor2d([inputData]);
            const predictionTensor = model.predict(tensor) as tf.Tensor;
            const predictedIndex = predictionTensor.argMax(1).dataSync()[0];
            
            // Map index to gesture (you would need a map or array here)
            const gestureMap = ['A', 'B', 'C', 'None'];
            setPrediction(gestureMap[predictedIndex] || 'Unknown');
            
            tensor.dispose();
            predictionTensor.dispose();
          }
        } else {
           setPrediction('No hands detected');
        }
      }
      
      animationFrameId = requestAnimationFrame(predictWebcam);
    };

    let lastVideoTime = -1;
    initializeMediaPipe();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (handLandmarker) handLandmarker.close();
    };
  }, [model]); // Need to wait for model to load, or not

  const drawLandmarks = (ctx: CanvasRenderingContext2D | null, landmarks: any[], width: number, height: number) => {
    if (!ctx) return;
    ctx.fillStyle = 'red';
    ctx.strokeStyle = 'blue';
    ctx.lineWidth = 2;
    for (const landmark of landmarks) {
      const x = landmark.x * width;
      const y = landmark.y * height;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
    // You can also draw the connections (skeleton) here
  };


  return (
    <div className="App" style={{ position: 'relative', width: '640px', height: '480px' }}>
      <h1>Sign Language Recognition</h1>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ position: 'absolute', top: 0, left: 0, width: '640px', height: '480px', transform: 'scaleX(-1)' }}
      />
      <canvas
        ref={canvasRef}
        width="640"
        height="480"
        style={{ position: 'absolute', top: 0, left: 0, zIndex: 10, transform: 'scaleX(-1)' }}
      />
      <div style={{ position: 'absolute', top: '490px', fontSize: '24px', fontWeight: 'bold' }}>
        Prediction: {prediction}
      </div>
    </div>
  );
}

export default App;

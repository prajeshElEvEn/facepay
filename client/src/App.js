import { useEffect, useRef } from "react";
import "./App.css";
import * as faceapi from "face-api.js";
import Webcam from "react-webcam";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const detect = async (labeledFaceDescriptors) => {
    // Check data is available
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set video width
      webcamRef.current.video.width = videoWidth;
      webcamRef.current.video.height = videoHeight;

      // Set canvas height and width
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
      const detections = await faceapi
        .detectAllFaces(video, new faceapi.SsdMobilenetv1Options())
        .withFaceLandmarks()
        .withFaceDescriptors();
      // console.log(detections);

      canvasRef.current.innerHTML = faceapi.createCanvasFromMedia(video);

      faceapi.matchDimensions(canvasRef.current, {
        width: videoWidth,
        height: videoHeight,
      });
      const resizedDetections = faceapi.resizeResults(detections, {
        width: videoWidth,
        height: videoHeight,
      });
      faceapi.draw.drawDetections(canvasRef.current, resizedDetections);
      const results = resizedDetections.map((d) =>
        faceMatcher.findBestMatch(d.descriptor)
      );

      results.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvasRef.current);
      });
    }
  };

  const recogFace = async () => {
    const labels = [
      "Black Widow",
      "Captain America",
      "Captain Marvel",
      "Hawkeye",
      "Jim Rhodes",
      "Thor",
      "Tony Stark",
    ];

    const labeledDescriptors = [];

    for (const label of labels) {
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/WebDevSimplified/Face-Recognition-JavaScript/master/labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      console.log("descriptions loaded");
      const labeledDescriptor = new faceapi.LabeledFaceDescriptors(
        label,
        descriptions
      );
      labeledDescriptors.push(labeledDescriptor);
    }

    return labeledDescriptors;
  };

  useEffect(() => {
    const loadNet = async () => {
      await faceapi.nets.ssdMobilenetv1.loadFromUri("/models");
      await faceapi.nets.ageGenderNet.loadFromUri("/models");
      await faceapi.nets.faceExpressionNet.loadFromUri("/models");
      await faceapi.nets.faceLandmark68Net.loadFromUri("/models");
      await faceapi.nets.faceRecognitionNet.loadFromUri("/models");
      console.log("Loaded models");
      const labeledFaceDescriptors = await recogFace();
      console.log(labeledFaceDescriptors);

      // recogFace();
      setInterval(async () => {
        await detect(labeledFaceDescriptors);
      }, 10);
    };

    loadNet();
  }, []);

  return (
    <>
      <Webcam
        ref={webcamRef}
        muted={true}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 9,
          width: 640,
          height: 480,
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          marginLeft: "auto",
          marginRight: "auto",
          left: 0,
          right: 0,
          textAlign: "center",
          zindex: 8,
          width: 640,
          height: 480,
        }}
      />
      {/* <button onClick={loadNet}>Load</button> */}
    </>
  );
}

export default App;

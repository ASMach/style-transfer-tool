import React, { useState } from "react";

import UploadButton from "./UploadButton";

import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";

import axios from "axios";

export default function Upload() {
  const [epochs, setEpochs] = useState(10);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);

  function targetImageCallback(data) {
    // Update the name in the component's state
    this.setState({ targetData: data });
  }

  function styleImageCallback(data) {
    // Update the name in the component's state
    this.setState({ styleData: data });
  }

  function handleEpochChange(e) {
    setEpochs(e.target.value);
  }

  function handleWidthChange(e) {
    setWidth(e.target.value);
  }

  function handleHeightChange(e) {
    setHeight(e.target.value);
  }

  function uploadImages(e) {
    e.preventDefault();
    console.log("Uploading images");

    var formData = new FormData();
    var targetImage = document.querySelector("#file");
    var styleImage = document.querySelector("#file");
    formData.append("target", targetImage.files[0]);
    formData.append("source", styleImage.files[0]);
    formData.append("epochs", epochs);
    formData.append("width", width);
    formData.append("height", height);

    axios
      .post("/transfer_style", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      })
      .then((response) => {
        console.log(response);
      })
      .catch((error) => {
        console.log(error);
      })
      .catch((error) => {
        if (error.response) {
          console.log(error.response);
          console.log(error.response.status);
          console.log(error.response.headers);
        }
      });
  }

  return (
    <main>
      <article>
        <h1>Welcome to Art Style Transfer Tool</h1>
        <p>
          Upload a picture and a style reference image, then specify how many
          epochs to train for and how big the image should be.
        </p>
      </article>
      <div>
        <Form onSubmit={uploadImages}>
          <Form.Label>Image</Form.Label>
          <UploadButton
            type="file"
            name="target"
            id="target"
            parentCallback={targetImageCallback}
          />
          <br />
          <Form.Label>Style</Form.Label>
          <UploadButton
            type="file"
            name="source"
            id="source"
            parentCallback={styleImageCallback}
          />
          <br />
          <Form.Label id="epochs">Epochs: {epochs}</Form.Label>
          <Form.Range
            value={epochs}
            min={1}
            max={100}
            onChange={handleEpochChange}
            step={1}
          />
          <Form.Label id="width">Width: {width}</Form.Label>
          <Form.Range
            value={width}
            min={128}
            max={2048}
            onChange={handleWidthChange}
            step={8}
          />
          <Form.Label id="height">Height: {height}</Form.Label>
          <Form.Range
            value={height}
            min={128}
            max={2048}
            onChange={handleHeightChange}
            step={8}
          />
          <Button variant="primary" type="submit">
            Submit
          </Button>
        </Form>
      </div>
    </main>
  );
}

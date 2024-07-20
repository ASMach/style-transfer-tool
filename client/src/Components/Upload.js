import React, { useState } from "react";

import UploadButton from "./UploadButton";

import Button from "react-bootstrap/Button";
import Form from "react-bootstrap/Form";

export default function Upload() {
  const [epochs, setEpochs] = useState(10);
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);

  function handleEpochChange(e) {
    setEpochs(e.target.value);
  }

  function handleWidthChange(e) {
    setWidth(e.target.value);
  }

  function handleHeightChange(e) {
    setHeight(e.target.value);
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
        <Form>
          <Form.Label>Image</Form.Label>
          <UploadButton type="file" name="target" id="target" />
          <br />
          <Form.Label>Style</Form.Label>
          <UploadButton type="file" name="source" id="source" />
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

import "./App.css";

import React from "react";

import Upload from "./Components/Upload";
import Files from "./Components/Files";

function App() {
  return (
    <div className="App">
      <Upload />
      <br />
      <Files />
    </div>
  );
}

export default App;

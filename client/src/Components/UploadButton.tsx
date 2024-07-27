import React, { useRef, useState } from "react";
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';


function UploadButton() {
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  
  const handleUpload = () => {
    if (inputRef.current && fileValidation(inputRef)) {
      inputRef.current?.click();
    }
  };
  const handleDisplayFileDetails = () => {
    inputRef.current?.files &&
      setUploadedFileName(inputRef.current.files[0].name);
  };


  const handleDisplayImage = () => {
    var reader = new FileReader();
    if (!inputRef.current?.files || !inputRef.current || !inputRef || !inputRef.current?.files[0]) {
      return (
      <div>
        No Image Selected
      </div>);
    }
    reader.readAsDataURL(inputRef.current?.files[0]);
    return (
      <div>
        <img src={URL.createObjectURL(inputRef.current?.files![0])}
        alt="img"
        className="preview"/>
      </div>
    );
  };
  const fileValidation = (e: React.RefObject<HTMLInputElement>) => {
  const filePath = e.current?.value;
 
  // Allowing file type
  const allowedExtensions = /(\.jpg|\.jpeg|\.png|\.gif)$/i;
    
  if (filePath && !allowedExtensions.exec(filePath)) {
    alert('Invalid file type');
    return false;
  }

  return true;
}
  return (
    <Form.Group className="m-3">
      <Form.Label className="mx-3">Choose file: </Form.Label>
      <input
        ref={inputRef}
        onChange={handleDisplayFileDetails}
        className="d-none"
        type="file"
      />
      <Button
        onClick={handleUpload}
        className={`btn btn-outline-${
          uploadedFileName ? "success" : "primary"
        }`}
      >
        {uploadedFileName ? uploadedFileName : "Upload"}
      </Button>
      {handleDisplayImage()}
    </Form.Group>
  );
}

export default UploadButton;
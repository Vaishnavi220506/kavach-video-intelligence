import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";

/* The generated @font-face sheet lives in public/ beside the woff2 files and
   is linked from index.html, so the browser can start fetching the faces
   before the module graph resolves. */
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/plate.css";
import "./styles/site.css";
import "./styles/workspace.css";
import "./styles/shell.css";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);

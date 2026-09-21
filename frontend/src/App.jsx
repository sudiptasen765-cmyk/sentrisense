import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout.jsx";
import Predict from "./pages/Predict.jsx";
import Compare from "./pages/Compare.jsx";
import Deployment from "./pages/Deployment.jsx";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Predict />} />
        <Route path="/compare" element={<Compare />} />
        <Route path="/deployment" element={<Deployment />} />
      </Route>
    </Routes>
  );
}

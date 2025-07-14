import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [userInput, setUserInput] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!userInput.trim()) return;

    const userMessage = { type: "user", text: userInput };
    setChatLog((prev) => [...prev, userMessage]);
    setUserInput("");
    setLoading(true);

    try {
      const encodedMessage = encodeURIComponent(userInput);
      const res = await axios.get(
        `https://210010fb523d.ngrok-free.app/query?message=${encodedMessage}`
      );

      const botMessage = { type: "bot", text: res.data.top.res };
      setChatLog((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
      setChatLog((prev) => [
        ...prev,
        {
          type: "bot",
          text: "⚠️ Something went wrong connecting to the bot.",
        },
      ]);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-6">
      <div className="bg-white p-8 rounded-2xl shadow-md w-full max-w-xl">
        <h1 className="text-2xl font-bold mb-4 text-center">
          Customised ChatBot
        </h1>

        <div className="bg-gray-50 p-4 rounded h-64 overflow-y-auto border mb-4">
          {chatLog.map((msg, index) => (
            <p
              key={index}
              className={`mb-2 ${
                msg.type === "bot" ? "text-blue-600" : "text-green-700"
              }`}
            >
              <strong>{msg.type === "bot" ? "Bot" : "You"}:</strong> {msg.text}
            </p>
          ))}
        </div>

        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Type your message..."
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;

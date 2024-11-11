document.getElementById("send-button").addEventListener("click", function () {
    const userMessage = document.getElementById("user-input").value;
    if (userMessage.trim()) {
        addMessageToChat("user", userMessage);
        document.getElementById("user-input").value = '';

        // Simulate bot response (replace this with actual logic)
        setTimeout(() => {
            addMessageToChat("bot", "This is a simulated response."); // Replace with actual bot logic
        }, 1000);
    }
});

document.getElementById("upload-button").addEventListener("click", function () {
    const fileInput = document.getElementById("pdf-upload");
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        uploadPDF(file);
    }
});

// Summary Button Event Listeners
document.getElementById("brief-summary-button").addEventListener("click", function () {
    const userMessage = document.getElementById("user-input").value;
    if (userMessage.trim()) {
        addMessageToChat("user", userMessage);
        getSummary(userMessage, "Brief");
    }
});

document.getElementById("detailed-summary-button").addEventListener("click", function () {
    const userMessage = document.getElementById("user-input").value;
    if (userMessage.trim()) {
        addMessageToChat("user", userMessage);
        getSummary(userMessage, "Detailed");
    }
});

document.getElementById("identify-problems-button").addEventListener("click", function () {
    const userMessage = document.getElementById("user-input").value;
    if (userMessage.trim()) {
        addMessageToChat("user", userMessage);
        identifyProblems(userMessage);
    }
});

document.getElementById("explain-problems-button").addEventListener("click", function () {
    const userMessage = document.getElementById("user-input").value;
    if (userMessage.trim()) {
        addMessageToChat("user", userMessage);
        explainProblems(userMessage);
    }
});

function uploadPDF(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.text) {
            addMessageToChat("bot", data.text);
        } else {
            addMessageToChat("bot", "Error processing PDF.");
        }
    });
}

function getSummary(message, type) {
    const language = document.getElementById("language-select").value;
    fetch('/summarize', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message, type: type, language: language })
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChat("bot", data.summary);
    });
}

function identifyProblems(message) {
    const language = document.getElementById("language-select").value;
    fetch('/identify_problems', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message, language: language })
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChat("bot", data.explanation);
    });
}

// New function to explain problems
function explainProblems(message) {
    const language = document.getElementById("language-select").value;
    fetch('/explain_problems', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message, language: language })
    })
    .then(response => response.json())
    .then(data => {
        addMessageToChat("bot", data.explanation);
    });
}

function addMessageToChat(sender, message) {
    const chatBody = document.getElementById("chat-body");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");
    messageDiv.textContent = message;
    chatBody.appendChild(messageDiv);
    chatBody.scrollTop = chatBody.scrollHeight; // Auto-scroll to the latest message
}

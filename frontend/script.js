const BACKEND_URL = window.BACKEND_URL || "http://127.0.0.1:8000";

const CATEGORY_MAP = {
  "Incoming Student": "incoming",
  "Current Student": "current",
  "Graduating Student": "graduating",
};

document.addEventListener("DOMContentLoaded", () => {
  const chatWindow = document.getElementById("chat-window");
  const form = document.getElementById("question-form");
  const input = document.getElementById("question");
  const categorySelect = document.getElementById("category");
  const submitButton = form.querySelector("button");

  const appendMessage = ({ text, sender, markdown = false, typing = false }) => {
    const message = document.createElement("div");
    message.className = `message message--${sender}`;
    if (typing) {
      message.classList.add("message--typing");
    }

    const content = document.createElement("div");
    if (markdown) {
      const rendered = marked.parse(text);
      content.innerHTML = DOMPurify.sanitize(rendered);
      content.querySelectorAll("a").forEach((link) => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
      });
    } else {
      content.textContent = text;
    }
    message.appendChild(content);
    chatWindow.appendChild(message);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    return message;
  };

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) {
      return;
    }

    const selectedLabel = categorySelect.value;
    const category = CATEGORY_MAP[selectedLabel] || "incoming";

    appendMessage({ text: question, sender: "user" });
    input.value = "";
    input.focus();
    submitButton.disabled = true;

    const typingMessage = appendMessage({
      text: "Advisor is typing…",
      sender: "bot",
      typing: true,
    });

    try {
      const response = await fetch(`${BACKEND_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, category }),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        const errorMessage =
          errorPayload.detail || "Something went wrong. Please try again.";
        typingMessage.remove();
        appendMessage({ text: errorMessage, sender: "bot" });
        return;
      }

      const data = await response.json();
      typingMessage.remove();
      appendMessage({
        text: data.answer || "No answer returned.",
        sender: "bot",
        markdown: true,
      });
    } catch (error) {
      console.error(error);
      typingMessage.remove();
      appendMessage({
        text: "Unable to contact the Advisor service right now. Please try again shortly.",
        sender: "bot",
      });
    } finally {
      submitButton.disabled = false;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }
  });
});

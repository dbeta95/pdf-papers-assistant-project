const submitButton = document.getElementById('submit');
const clearButton = document.getElementById('clear');
const questionInput = document.getElementById('question');
const categorySelect = document.getElementById('category');
const answerDiv = document.getElementById('answer');
const loadingDiv = document.getElementById('loading');
const thumbsUpButton = document.getElementById('thumbsUp');
const thumbsDownButton = document.getElementById('thumbsDown');
const feedbackMessage = document.getElementById('feedbackMessage')

submitButton.addEventListener('click', async () => {
    const question = questionInput.value;
    const category = categorySelect.value;

    // Show loading animation
    loadingDiv.style.display = 'block';
    answerDiv.textContent = '';

    try {
        const data = category ? { question, category } : { question };
        const response = await fetch('http://127.0.0.1:5000/question', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        answerDiv.textContent = result.answer;

        // Add event listeners for feedback buttons
        thumbsUpButton.addEventListener('click', () => {
            sendFeedback(result.conversation_id, 1);
        });

        thumbsDownButton.addEventListener('click', () => {
            sendFeedback(result.conversation_id, -1);
        });

    } catch (error) {
        console.error('Error fetching answer:', error);
        answerDiv.textContent = 'Error fetching answer. Please try again later.';
    } finally {
        // Hide loading animation
        loadingDiv.style.display = 'none';
    }
});

clearButton.addEventListener('click', () => {
    questionInput.value = '';
    categorySelect.selectedIndex = 0;
    answerDiv.textContent = '';
});

async function sendFeedback(conversationId, feedback) {
    try {
        const response = await fetch('http://127.0.0.1:5000/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ conversation_id: conversationId, feedback })
        });

        // Handle the response from the feedback endpoint if needed
        console.log('Feedback sent:', response);

        // Display feedback message
        feedbackMessage.textContent = "Feedback saved, thank you!";
        setTimeout(() => {
            feedbackMessage.textContent = ""; // Clear the message after a few seconds
        }, 3000); 

    } catch (error) {
        console.error('Error sending feedback:', error);
    }
}
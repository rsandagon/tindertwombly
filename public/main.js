// actions
let messageBtn = document.getElementById('messageBtn');
let messageInput = document.getElementById('messageInput');
let previousMessage = "";
let history = [];
let personality_raw = [];

addPreviousMessage = (message)=>{
    history.push(message)

    if(history.length >  6){
        history.shift()
    }
}

messageBtn.onclick = (evnt) => {
    addMyMessage(messageInput.value);
    messageInput.value = "";
}

messageInput.addEventListener("keyup", function (event) {
    if (event.keyCode === 13) {
        event.preventDefault();
        messageBtn.click();
    }
});

// methods
showTypingIndicator = (bol) => {
    let indicator = document.getElementById('typingIndicator')
    bol ? indicator.classList.remove('hidden') : indicator.classList.add('hidden');
}

addTwomblyMessage = (message) => {
    let cbox = document.getElementById('chatbox');
    let d = new Date();
    addPreviousMessage(message)
    let myMessageTemplate = `<div class="flex items-start mb-8 text-sm">
<div class="bg-cover rounded-full h-10 w-10 flex items-center justify-center mr-4" style="background-image: url(https://source.unsplash.com/fdCvrdYUJsY)"></div>
<div class="flex-1 overflow-hidden">
    <div class="text-left">
        <span class="font-bold text-l text-red-600 title">Twombly</span>
        <span class="text-grey text-l">${d.getHours()}:${d.getMinutes()}</span>
    </div>
    <p class="text-xl text-left">${message}</p>
</div>
</div>`
    //optimize with appendChild instead
    cbox.innerHTML += myMessageTemplate;
    showTypingIndicator(false);
}

addMyMessage = (message) => {
    //addPreviousMessage(message)
    console.log('added ', message)
    let cboxContainer = document.getElementById('chatboxContainer');
    let cbox = document.getElementById('chatbox');
    let quoute = document.getElementById('blockQuote');
    let d = new Date();

    let myMessageTemplate = `<div class="flex items-start mb-8 text-sm">
<div class="bg-cover rounded-full h-10 w-10 flex items-center justify-center mr-4" style="background-image: url(https://source.unsplash.com/BgRs4dzW4Js)"></div>
<div class="flex-1 overflow-hidden">
    <div class="text-left">
        <span class="font-bold text-l title">Me</span>
        <span class="text-grey text-l">${d.getHours()}:${d.getMinutes()}</span>
    </div>
    <p class="text-xl text-left">${message}</p>
</div>
</div>`
    //optimize with appendChild instead
    cbox.innerHTML += myMessageTemplate;
    showTypingIndicator(true);

    sendChat(message);

    // quoute.classList.add('hidden');
    cboxContainer.scrollTop = cboxContainer.scrollHeight;
}

//init
window.onload = ()=>{
    fetch('/api')
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);
            let quoute = document.getElementById('blockQuote');
            quoute.innerText =`“${data.personality}“`;
            personality_raw = data.personality_raw;
        })
        .catch((error) => {
            console.warn('Error:', error);
        });
}



//fetch
sendChat = (message) => {
    let payload = {
        "raw": message,
        "personality": "string",
        "personality_raw": personality_raw,
        "reply": "",
        "history":history,
        "top_k":0,
        "top_p":0.9,
        "temperature":0.7   
    }


    // //med
    // top_k:70;
    // top_p:0.5;
    // temperature:1.2;
    // //high
    // top_k:0;
    // top_p:0.9;
    // temperature:0.6;
    // //low
    // top_k:180;
    // top_p:180;
    // temperature:1.9;

    fetch('/api/chat', {
        method: 'POST', // or 'PUT'
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    })
        .then((response) => response.json())
        .then((data) => {
            console.log('Success:', data);
            addTwomblyMessage(data.reply);
        })
        .catch((error) => {
            console.warn('Error:', error);
        });

}

//Init tooltips
tippy('.link', {
    placement: 'bottom'
})

//Toggle mode
const toggle = document.querySelector('.js-change-theme');
const body = document.querySelector('body');
const profile = document.getElementById('profile');


toggle.addEventListener('click', () => {

    if (body.classList.contains('text-gray-900')) {
        toggle.innerHTML = "☀️";
        body.classList.remove('text-gray-900');
        body.classList.add('text-gray-100');
        profile.classList.remove('bg-white');
        profile.classList.add('bg-gray-900');
    } else {
        toggle.innerHTML = "🌙";
        body.classList.remove('text-gray-100');
        body.classList.add('text-gray-900');
        profile.classList.remove('bg-gray-900');
        profile.classList.add('bg-white');

    }
});


//initialize Twombly
addTwomblyMessage('Hello!')
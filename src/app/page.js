'use client'

import { useState, useRef, useEffect } from "react";
import { Box, TextField, Button, Stack, Typography, Paper, IconButton } from "@mui/material";
import SendIcon from '@mui/icons-material/Send';
import { ThemeProvider, createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#388e3c',
    },
  },
});

export default function Home() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hello, I am the Rate my professor support system. How can I help you today?",
    }
  ]);
  const [message, setMessage] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const sendMessage = async () => {
    if (message.trim() === '') return;

    setMessages((prevMessages) => [
      ...prevMessages,
      { role: "user", content: message },
      { role: "assistant", content: '' }
    ]);
    setMessage('');

    try {
      const response = await fetch('/api/chat', {
        method: "POST",
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify([...messages, { role: "user", content: message }])
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let result = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        result += text;
        setMessages((prevMessages) => {
          const lastMessage = prevMessages[prevMessages.length - 1];
          return [
            ...prevMessages.slice(0, -1),
            { ...lastMessage, content: lastMessage.content + text },
          ];
        });
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box
        sx={{
          width: "100vw",
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          bgcolor: '#f5f5f5',
        }}
      >
        <Paper
          elevation={3}
          sx={{
            width: "90%",
            maxWidth: "600px",
            height: "80vh",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Typography variant="h5" sx={{ p: 2, borderBottom: '1px solid #e0e0e0', bgcolor: 'primary.main', color: 'white' }}>
            Rate My Professor Chat
          </Typography>
          <Stack
            direction="column"
            spacing={2}
            sx={{
              p: 2,
              overflow: "auto",
              flexGrow: 1,
            }}
          >
            {messages.map((message, index) => (
              <Box
                key={index}
                sx={{
                  display: "flex",
                  justifyContent: message.role === "assistant" ? 'flex-start' : 'flex-end',
                }}
              >
                <Paper
                  elevation={1}
                  sx={{
                    bgcolor: message.role === "assistant" ? "primary.light" : "secondary.light",
                    color: 'text.primary',
                    borderRadius: 2,
                    p: 1.5,
                    maxWidth: '70%',
                  }}
                >
                  <Typography variant="body1">
                    {message.content}
                  </Typography>
                </Paper>
              </Box>
            ))}
            <div ref={messagesEndRef} />
          </Stack>
          <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0' }}>
            <Stack direction="row" spacing={1}>
              <TextField
                label="Type your message"
                fullWidth
                variant="outlined"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                sx={{ bgcolor: 'white' }}
              />
              <IconButton
                color="primary"
                onClick={sendMessage}
                disabled={message.trim() === ''}
                sx={{ bgcolor: 'primary.main', color: 'white', '&:hover': { bgcolor: 'primary.dark' } }}
              >
                <SendIcon />
              </IconButton>
            </Stack>
          </Box>
        </Paper> 
      </Box>
    </ThemeProvider>
  );
}
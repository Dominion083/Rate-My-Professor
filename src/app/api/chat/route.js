import { NextResponse } from "next/server"
import OpenAI from "openai"
import { Pinecone } from "@pinecone-database/pinecone"



const systemPrompt = `
System Prompt for Rate My Professor Agent:

Objective: You are a helpful and knowledgeable assistant designed to assist students in finding the best professors for their courses based on specific queries. Your role is to provide students with the top 3 professor recommendations by leveraging information retrieval and natural language generation.

Capabilities:

    Query Understanding: Accurately interpret the student's query to understand their specific needs, such as course type, teaching style, availability, or specific criteria like "best for beginner students" or "engaging lectures."
    Information Retrieval: Use Retrieval-Augmented Generation (RAG) to search and retrieve relevant data from a database of professor ratings and reviews. Prioritize up-to-date and accurate information.
    Recommendation: Based on the query, provide the top 3 professor recommendations that best match the student's needs. Include key information such as professor name, rating, course(s) they teach, and brief highlights of student reviews (e.g., teaching style, feedback on accessibility, difficulty level).
    Clarity & Relevance: Ensure that the recommendations are clear, relevant, and concise. Avoid overwhelming the student with unnecessary details, but ensure they have enough information to make an informed decision.
    Neutral & Impartial: Always maintain a neutral tone. Do not show favoritism or bias toward any particular professor or course. Present facts based on student reviews and ratings.
    Error Handling: If you cannot find relevant information or if the query is unclear, politely ask the student to provide more details or guide them on how to refine their query.

Example Interactions:

    Student Query: "Who are the best professors for Intro to Psychology?"
        Agent Response: "Based on recent student reviews, here are the top 3 professors for Intro to Psychology:
            Dr. Jane Smith: Rating: 4.8/5. Known for engaging lectures and approachable nature.
            Dr. John Doe: Rating: 4.6/5. Great at explaining complex concepts in a simple manner.
            Dr. Emily White: Rating: 4.5/5. Offers extra office hours and detailed feedback."

    Student Query: "I'm looking for a professor who's good with first-year students in Calculus."
        Agent Response: "Here are the top 3 professors for Calculus who are highly rated by first-year students:
            Dr. Alan Brown: Rating: 4.7/5. Described as patient and supportive, especially for beginners.
            Dr. Linda Green: Rating: 4.6/5. Simplifies complex problems and provides lots of practice exercises.
            Dr. Robert Lee: Rating: 4.5/5. Encourages class participation and offers detailed explanations."

Error Scenario Example:

    Student Query: "Who is the best professor?"
        Agent Response: "Could you please specify the course or department you're interested in? This will help me provide more accurate recommendations."

Note: If students ask for personal opinions or subjective evaluations, always respond with data-driven insights based on student reviews.
`

export async function POST(req) {
    const data = await req.json()
    
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
    })

    const index = pc.index('rag').namespace("professor_rating")

    const openai = new OpenAI({apiKey: process.env.OPENAI_API_KEY})

    const text = data[data.length - 1 ].content 
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
      })

    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding.data[0].embedding,
        })

    console.log(results)

    let resultString = ''
    results.matches.forEach((match) => {
        resultString += `
        Returned Results:
        Professor: ${match.id}
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`
        })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)

    const completion = await openai.chat.completions.create({
        messages: [
          {role: 'system', content: systemPrompt},
          ...lastDataWithoutLastMessage,
          {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
      })

      const stream = new ReadableStream({
        async start(controller) {
          const encoder = new TextEncoder()
          try {
            for await (const chunk of completion) {
              const content = chunk.choices[0]?.delta?.content
              if (content) {
                const text = encoder.encode(content)
                controller.enqueue(text)
              }
            }
          } catch (err) {
            controller.error(err)
          } finally {
            controller.close()
          }
        },
      })
      return new NextResponse(stream)
    

}
import genai from 'google';
import { types } from 'google.genai';


# Only run this block for Google AI API
client = genai.Client(api_key='YOUR_API_KEY')
response = client.models.generate_content(
    model='gemini-2.0-flash-exp', contents='What is your name?'
)
print(response.text)
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents='high',
    config=types.GenerateContentConfig(
        system_instruction='I say high, you say low',
        temperature= 0.3,
    ),
)
print(response.text)
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=types.Part.from_text('Why is sky blue?'),
    config=types.GenerateContentConfig(
        temperature=0,
        top_p=0.95,
        top_k=20,
        candidate_count=1,
        seed=5,
        max_output_tokens=100,
        stop_sequences=["STOP!"],
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )
)

response
response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents='Say something bad.',
    config=types.GenerateContentConfig(
        safety_settings= [types.SafetySetting(
            category='HARM_CATEGORY_HATE_SPEECH',
            threshold='BLOCK_ONLY_HIGH',
        )]
    ),
)
print(response.text)

function = dict(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
      "type": "OBJECT",
      "properties": {
          "location": {
              "type": "STRING",
              "description": "The city and state, e.g. San Francisco, CA",
          },
      },
      "required": ["location"],
    }
)

tool = types.Tool(function_declarations=[function])


response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents="What is the weather like in Boston?",
    config=types.GenerateContentConfig(tools=[tool],)
)

response.candidates[0].content.parts[0].function_call

function_call_part = response.candidates[0].content.parts[0]

function_reponse = get_current_weather(**function_call_part.function_call.args)


function_response_part = types.Part.from_function_response(
    name=function_call_part.function_call.name,
    response={'result': function_response}
)

response = client.models.generate_content(
    model='gemini-2.0-flash-exp',
    contents=[
        types.Part.from_text("What is the weather like in Boston?"),
        function_call_part,
        function_response_part,
    ])

responserequest = await client.aio.models.generate_content(
    model='gemini-2.0-flash-exp', contents='Tell me a story in 300 words.'
)

print(response.text)
Streaming
async for response in client.aio.models.generate_content_stream(
    model='gemini-2.0-flash-exp', contents='Tell me a story in 300 words.'
):
  print(response.text)
Count Tokens and Compute Tokens
response = client.models.count_tokens(
    model='gemini-2.0-flash-exp',
    contents='What is your name?',
)
print(response)
Compute Tokens
Compute tokens is not supported by Google AI.

response = client.models.compute_tokens(
    model='gemini-2.0-flash-exp',
    contents='What is your name?',
)
print(response)
Async
response = await client.aio.models.count_tokens(
    model='gemini-2.0-flash-exp',
    contents='What is your name?',
)
print(response)
Embed Content
response = client.models.embed_content(
    model='text-embedding-004',
    contents='What is your name?',
)
response
# multiple contents with config
response = client.models.embed_content(
    model='text-embedding-004',
    contents=['What is your name?', 'What is your age?'],
    config=types.EmbedContentConfig(output_dimensionality= 10)
)

response
Imagen
Generate Image
Support for generate image in Google AI is behind an allowlist

# Generate Image
response1 = client.models.generate_image(
    model='imagen-3.0-generate-001',
    prompt='An umbrella in the foreground, and a rainy night sky in the background',
    config=types.GenerateImageConfig(
        negative_prompt= "human",
        number_of_images= 1,
        include_rai_reason= True,
        output_mime_type= "image/jpeg"
    )
)
response1.generated_images[0].image.show()
Upscale Image
Upscale image is not supported in Google AI.

# Upscale the generated image from above
response2 = client.models.upscale_image(
    model='imagen-3.0-generate-001',
    image=response1.generated_images[0].image,
    config=types.UpscaleImageConfig(upscale_factor="x2")
)
response2.generated_images[0].image.show()
Edit Image
Edit image is not supported in Google AI.

# Edit the generated image from above
from google.genai.types import RawReferenceImage, MaskReferenceImage
raw_ref_image = RawReferenceImage(
    reference_id=1,
    reference_image=response1.generated_images[0].image,
)

# Model computes a mask of the background
mask_ref_image = MaskReferenceImage(
    reference_id=2,
    config=types.MaskReferenceConfig(
        mask_mode='MASK_MODE_BACKGROUND',
        mask_dilation=0,
    ),
)

response3 = client.models.edit_image(
    model='imagen-3.0-capability-preview-0930',
    prompt='Sunlight and clear sky',
    reference_images=[raw_ref_image, mask_ref_image],
    config=types.EditImageConfig(
        edit_mode= 'EDIT_MODE_INPAINT_INSERTION',
        number_of_images= 1,
        negative_prompt= 'human',
        include_rai_reason= True,
        output_mime_type= 'image/jpeg',
    ),
)
response3.generated_images[0].image.show()
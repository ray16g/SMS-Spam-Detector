import React from 'react'
import './index.css'

function getColor(res)
{
  if(res > 0)
    return 'IndianRed'
  else if(res < 0)
    return 'LimeGreen'
  else
    return 'Silver'
}

function getClass(res)
{
  if(res > 0)
    return 'Spam'
  else if(res < 0)
    return 'Not Spam'
  else
    return 'Inconclusive'
}

const Info = ({data, setBack, text}) => {

  function handleClick(e)
  {
    e.preventDefault()
    setBack()
  }

  return (
    <div className='info'>

      <p className='mid'>The spam classifier determined that the text message is:</p>
      <span className='classify' style={{color: getColor(data.spam)}}>{getClass(data.spam)}</span>
      <p className='mid'>but why?</p>
      <p>Before the text gets put through the model, the first thing to do is convert the incoming text to a format readable by code. 
        This includes a couple of preprocesing steps:</p>
      <ul className='preprocessing-list'>
        <li>Tokenization: Breaking down the incoming words into indiviudal words for the program to read.</li>
        <li>Text Cleaning: Removing punctuation, whitespace, and converting all characters to lowercase.</li>
        <li>Removing Stop Words: Removing common words that does not add a lot of information such as "the", "and", and "a."</li>
        <li>Lemmatization: Converting words to their simpler forms to normalize text. For example, words "spamming", "spammed", "spammer", and "spam" to "spam</li>
      </ul>
      <p>Doing so converts our text from</p>
      <div className="mid bold">
      {
        text
      }
      </div>
      <p>to</p>
      <div className="mid bold">
      {
        data.text.map((word) => {
          return <span>{word} </span>
        })
      }
      </div>
      
      <p>After preprocessing, we put the input through a model. The model here is implementing used a <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier" target="_blank">Naive Bayes classifier</a>, which computes the probability of the input being spam versus not spam considering which words do not appear and which words appear. The words that make the input more likely to be spam are marked in red and the words that do not are marked in green below (gray means word is not in data):</p>
      <div className="mid bold">
      {
        data.text.map((word, i) => {
          return <span style={{color: getColor(data.class[i])}}>{word} </span>
      })
      }
      </div>
      <p>After comparing the probabilities, the model comes to a conclusion. Remember, the model also takes into account words that do not appear and certain words may have heavier weights than other words.</p>
      <button className='btn mid' onClick={handleClick}>Back</button>
    </div>
  )
}

export default Info